# src/ui_analyze.py
from __future__ import annotations

import base64
import hashlib
import io
import logging

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas

from .i18n import t
from .state import goto_step, reset_shot, reset_cv_cache
from .metrics import compute_metrics, classify_shape
from .rules import next_end_advice
from .storage import make_log_entry, export_log_json
from .cv_target import rectify_target, transform_points, propose_hit_points
from .refine_points import sample_contact_color_hsv
from .scoring import score_hits_color_aware
from .target_face import render_target_face_bgr, TARGET_FACES
from .coach import CoachRAG, CoachConfig

CANON_SIZE = 900
CANON_CENTER = (CANON_SIZE / 2.0, CANON_SIZE / 2.0)
CANON_OUTER = CANON_SIZE * 0.45  # 405 px
CANVAS_SIZE = 700
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_DIMENSION = 2400

logger = logging.getLogger(__name__)

SELF_REPORT_OPTIONS = [
    "none",
    "bow_hand_tense",
    "anchor_unclear",
    "release_forced",
    "aim_too_long",
    "bow_arm_dropped",
    "fatigue_or_pain",
    "wind_or_equipment",
]


def _bgr_to_rgb_uint8(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)


def _image_data_url(image_rgb: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image_rgb).save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _points_to_initial_drawing(points, r=10, background_rgb: np.ndarray | None = None):
    objects = []
    if background_rgb is not None:
        height, width = background_rgb.shape[:2]
        objects.append(
            {
                "type": "image",
                "version": "4.4.0",
                "originX": "left",
                "originY": "top",
                "left": 0,
                "top": 0,
                "width": int(width),
                "height": int(height),
                "scaleX": 1,
                "scaleY": 1,
                "angle": 0,
                "opacity": 1,
                "selectable": False,
                "evented": False,
                "hasControls": False,
                "hasBorders": False,
                "src": _image_data_url(background_rgb),
                "filters": [],
            }
        )
    for (x, y) in points:
        objects.append(
            {
                "type": "circle",
                "left": float(x - r),
                "top": float(y - r),
                "radius": float(r),
                "fill": "rgba(180, 0, 255, 0.22)",
                "stroke": "rgba(180, 0, 255, 0.95)",
                "strokeWidth": 2,
            }
        )
    return {"version": "4.4.0", "objects": objects}


def _extract_points_from_canvas(json_data: dict):
    if not json_data or "objects" not in json_data or not json_data["objects"]:
        return []
    pts = []
    for obj in json_data["objects"]:
        if obj.get("type") == "circle":
            left = float(obj.get("left", 0))
            top = float(obj.get("top", 0))
            r = float(obj.get("radius", 0))
            scale_x = float(obj.get("scaleX", 1.0) or 1.0)
            scale_y = float(obj.get("scaleY", 1.0) or 1.0)
            x = left if obj.get("originX") == "center" else left + r * scale_x
            y = top if obj.get("originY") == "center" else top + r * scale_y
            pts.append({"x": x, "y": y})
    return pts


def _scale_points(points: list[dict], factor: float) -> list[dict]:
    return [
        {"x": float(point["x"]) * factor, "y": float(point["y"]) * factor}
        for point in points
    ]


def _sanitize_canonical_points(points) -> list[dict]:
    sanitized = []
    for point in points or []:
        if isinstance(point, dict):
            x, y = point.get("x"), point.get("y")
        else:
            try:
                x, y = point
            except (TypeError, ValueError):
                continue

        try:
            x, y = float(x), float(y)
        except (TypeError, ValueError):
            continue

        if np.isfinite(x) and np.isfinite(y) and 0 <= x <= CANON_SIZE and 0 <= y <= CANON_SIZE:
            sanitized.append({"x": x, "y": y})
    return sanitized


def _update_points_from_canvas(current_points: list[dict], json_data: dict | None) -> list[dict]:
    """Persist canonical points across Streamlit's transient component reruns."""
    canvas_points = _extract_points_from_canvas(json_data)
    if not canvas_points:
        return list(current_points)
    # The drawable-canvas component can briefly emit its previous payload when
    # another Streamlit widget opens or changes. The toolbar has no delete
    # action in this app, so a smaller payload is stale rather than intentional.
    # Point removal is handled explicitly by the Clear button.
    if len(canvas_points) < len(current_points):
        return list(current_points)
    canonical = _scale_points(canvas_points, CANON_SIZE / CANVAS_SIZE)
    return _sanitize_canonical_points(canonical)


def _draw_hits_on_face(face_bgr, points_xy, scores):
    img = face_bgr.copy()
    PURPLE = (255, 0, 255)  # BGR
    for i, (x, y) in enumerate(points_xy):
        px, py = int(round(x)), int(round(y))
        cv2.circle(img, (px, py), 10, PURPLE, 2)
        cv2.circle(img, (px, py), 2, PURPLE, -1)
        if i < len(scores):
            s = scores[i]
            cv2.putText(
                img,
                str(s),
                (px + 12, py - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
    return img


def _canon_to_rect_points(points_xy, M_rect_to_canon):
    if not points_xy:
        return []
    if M_rect_to_canon is None:
        raise TypeError("M_rect_to_canon is None (cannot invert affine transform).")
    if not isinstance(M_rect_to_canon, np.ndarray):
        M_rect_to_canon = np.array(M_rect_to_canon, dtype=np.float32)
    if M_rect_to_canon.shape != (2, 3):
        raise TypeError(f"M_rect_to_canon must be shape (2,3), got {M_rect_to_canon.shape}")
    Minv = cv2.invertAffineTransform(M_rect_to_canon.astype(np.float32))
    pts = np.array(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.transform(pts, Minv).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in out]


def _read_image_from_uploader_or_camera() -> tuple[np.ndarray | None, str | None]:
    """
    Returns (img_rgb, cache_key) or (None, None)
    """
    lang = st.session_state.language

    # input mode selector
    st.session_state.image_mode = st.radio(
        t("input_mode", lang),
        options=["upload", "camera"],
        horizontal=True,
        index=0 if st.session_state.image_mode == "upload" else 1,
        format_func=lambda x: t("mode_upload", lang) if x == "upload" else t("mode_camera", lang),
    )

    if st.session_state.image_mode == "upload":
        st.subheader(t("upload", lang))
        file = st.file_uploader(
            t("upload", lang),
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
        )
        if not file:
            return None, None
        if getattr(file, "size", 0) > MAX_UPLOAD_BYTES:
            st.error(t("image_too_large", lang))
            return None, "error"
        return _decode_image(file, source="upload", lang=lang)

    # camera mode
    st.subheader(t("camera", lang))
    cam = st.camera_input(t("camera", lang), label_visibility="collapsed")
    if not cam:
        return None, None
    return _decode_image(cam, source="camera", lang=lang)


def _decode_image(file, source: str, lang: str) -> tuple[np.ndarray | None, str | None]:
    """Decode, orient, and bound an uploaded image before running OpenCV."""
    try:
        image = Image.open(file)
        image.load()
        image = ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        st.error(t("invalid_image", lang))
        return None, "error"

    if max(image.size) > MAX_IMAGE_DIMENSION:
        image.thumbnail(
            (MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION),
            Image.Resampling.LANCZOS,
        )

    img_rgb = np.asarray(image, dtype=np.uint8).copy()
    digest = hashlib.sha256(img_rgb.tobytes()).hexdigest()[:16]
    cache_key = f"{source}-{digest}-{img_rgb.shape[0]}x{img_rgb.shape[1]}"
    return img_rgb, cache_key


def render_analyze_step():
    lang = st.session_state.language
    st.title(t("title", lang))

    # -------------------------
    # Top controls
    # -------------------------
    top1, top2, top3 = st.columns([1, 1, 1.7])
    with top1:
        st.session_state.distance_m = st.number_input(
            t("distance", lang), min_value=3, max_value=90, value=int(st.session_state.distance_m), step=1
        )
    with top2:
        st.session_state.arrows_per_end = st.number_input(
            t("arrows", lang), min_value=3, max_value=12, value=int(st.session_state.arrows_per_end), step=1
        )
    with top3:
        face_keys = list(TARGET_FACES.keys())
        if st.session_state.target_face not in face_keys:
            st.session_state.target_face = face_keys[0]

        face = st.selectbox(
            t("target_face", lang),
            options=face_keys,
            index=face_keys.index(st.session_state.target_face),
            format_func=lambda k: t(TARGET_FACES[k]["label_key"], lang),
        )
        if face != st.session_state.target_face:
            st.session_state.target_face = face
            reset_shot()
            reset_cv_cache()
            st.session_state.canvas_revision += 1

    # -------------------------
    # Profile + Coach settings
    # -------------------------
    with st.expander(t("profile", lang), expanded=False):
        p = st.session_state.user_profile

        p["name"] = st.text_input(t("profile_name", lang), value=p.get("name", ""))
        p["bow"] = st.selectbox(t("profile_bow", lang), options=["recurve", "compound", "barebow"], index=["recurve", "compound", "barebow"].index(p.get("bow", "recurve")))
        p["experience_months"] = st.number_input(t("profile_exp", lang), min_value=0, max_value=600, value=int(p.get("experience_months", 0)), step=1)
        p["dominant_eye"] = st.text_input(t("profile_eye", lang), value=p.get("dominant_eye", ""))
        p["goals"] = st.text_area(t("profile_goals", lang), value=p.get("goals", ""), height=80)
        p["recurring_issues"] = st.text_area(t("profile_issues", lang), value=p.get("recurring_issues", ""), height=80)
        p["constraints"] = st.text_area(t("profile_constraints", lang), value=p.get("constraints", ""), height=80)

        style_map = {"tight": "style_tight", "gentle": "style_gentle", "technical": "style_technical"}
        style_keys = list(style_map.keys())
        p["language_style"] = st.selectbox(
            t("profile_style", lang),
            options=style_keys,
            index=style_keys.index(p.get("language_style", "tight")),
            format_func=lambda k: t(style_map[k], lang),
        )
        st.session_state.user_profile = p

    with st.expander(t("coach_settings", lang), expanded=False):
        # Migrate sessions created before book-guided coaching became the
        # default. Old RAG modes required unavailable cloud dependencies.
        if st.session_state.coach_mode not in {"book", "rules"}:
            st.session_state.coach_mode = "book"
        st.session_state.coach_mode = st.selectbox(
            t("coach_mode", lang),
            options=["book", "rules"],
            index=["book", "rules"].index(st.session_state.coach_mode),
            format_func=lambda key: t(f"coach_{key}", lang),
        )
        if st.session_state.coach_mode == "book":
            st.caption(t("coach_book_caption", lang))

    # -------------------------
    # Nav buttons
    # -------------------------
    colA, colB = st.columns([1, 1])
    with colA:
        if st.button(t("back", lang), use_container_width=True):
            goto_step("handedness")
            st.rerun()
    with colB:
        if st.button(t("clear", lang), use_container_width=True):
            reset_shot()
            st.session_state.canvas_revision += 1

    st.divider()

    # -------------------------
    # Upload / Camera
    # -------------------------
    img_rgb, cache_key = _read_image_from_uploader_or_camera()
    if img_rgb is None:
        if cache_key is None:
            st.info(t("workflow_help", lang))
        return

    need = int(st.session_state.arrows_per_end)

    # -------------------------
    # CV pipeline (cached)
    # -------------------------
    if st.session_state.get("cv_cache_key") != cache_key:
        try:
            with st.spinner(t("processing_photo", lang)):
                rect_res = rectify_target(img_rgb, out_size=CANON_SIZE)

                face_bgr = render_target_face_bgr(
                    st.session_state.target_face,
                    size=CANON_SIZE,
                    center=CANON_CENTER,
                    outer_radius=CANON_OUTER,
                    draw_ring_lines=True,
                    ring_line_thickness=2,
                )

                proposal_debug = {}
                coarse = propose_hit_points(
                    rect_res.rect_bgr,
                    rect_res.center_final,
                    rect_res.arrow_present,
                    max_points=max(need, 12),
                    outer_radius=rect_res.outer_radius,
                    diagnostics=proposal_debug,
                    quality_flags=rect_res.quality_flags,
                )
                rect_res.debug["hit_detection"] = proposal_debug

                detected_count = None
                if proposal_debug.get("mode") == "visible_shafts" and 3 <= len(coarse) <= 12:
                    detected_count = len(coarse)
                    # Visible shaft endpoints are direct current-arrow
                    # evidence, so use their count instead of padding the end
                    # with historical holes to match the previous setting.
                    need = detected_count
                    st.session_state.arrows_per_end = detected_count

                # Radial-anomaly candidates are already weighted centroids or
                # shaft endpoints. A second generic edge search can snap them
                # back to a nearby printed ring, so map them directly.
                auto_pts_canon = transform_points(coarse, rect_res.M_rect_to_canon)

            st.session_state.cv_cache_key = cache_key
            st.session_state.overlay_image_rgb = _bgr_to_rgb_uint8(face_bgr)
            st.session_state.auto_points = auto_pts_canon
            st.session_state.points = _sanitize_canonical_points(auto_pts_canon[:need])
            st.session_state.last_result = None
            st.session_state.end_self_report = "none"
            st.session_state.canvas_revision += 1

            st.session_state._geom_center = CANON_CENTER
            st.session_state._geom_outer = CANON_OUTER
            st.session_state._rect_photo_bgr = rect_res.rect_bgr.copy()
            st.session_state._M_rect_to_canon = rect_res.M_rect_to_canon.copy()
            st.session_state.warp_debug = rect_res.debug
            st.session_state.cv_quality = {
                "score": float(rect_res.quality_score),
                "flags": list(rect_res.quality_flags),
            }
            st.session_state.cv_detection_mode = proposal_debug.get("mode")
            st.session_state.detected_arrow_count = detected_count
        except Exception:
            logger.exception("Target photo analysis failed")
            reset_shot()
            reset_cv_cache()
            st.error(t("cv_error", lang))
            st.caption(t("cv_error_hint", lang))
            return

    if (
        st.session_state.get("cv_detection_mode") == "visible_shafts"
        and st.session_state.get("detected_arrow_count")
    ):
        need = int(st.session_state.detected_arrow_count)

    bg_rgb = st.session_state.overlay_image_rgb
    auto_pts = st.session_state.auto_points
    center = st.session_state._geom_center
    outer = st.session_state._geom_outer
    quality = st.session_state.get("cv_quality", None)

    with st.expander(t("source_preview", lang), expanded=False):
        st.image(img_rgb, caption=t("source_photo", lang), use_container_width=True)

    st.subheader(t("tap_points", lang))
    st.caption(t("mapped_target_hint", lang))
    quality_flags = set((quality or {}).get("flags", []) or [])
    if st.session_state.get("cv_detection_mode") == "visible_shafts":
        st.info(
            t("detected_arrow_count", lang).format(
                count=int(st.session_state.detected_arrow_count),
            )
        )
    if "image_blur" in quality_flags:
        st.warning(t("blur_warning", lang))
    elif {"image_low_light", "image_glare"}.intersection(quality_flags):
        st.warning(t("exposure_warning", lang))
    elif "image_low_contrast" in quality_flags:
        st.warning(t("contrast_warning", lang))
    elif quality is not None and float(quality.get("score", 1.0)) < 0.55:
        st.warning(t("quality_warning", lang))
    if not auto_pts:
        if st.session_state.get("cv_detection_mode") not in {
            "blur_rejected",
            "exposure_rejected",
        }:
            st.info(t("manual_points", lang))
    elif len(st.session_state.points) < need:
        st.info(
            t("partial_points", lang).format(
                count=len(st.session_state.points),
                need=need,
            )
        )
    with st.expander(t("metric_debug", lang)):
        st.write(st.session_state.warp_debug)
        if quality is not None:
            st.write({"quality": quality})

    canonical_points = _sanitize_canonical_points(st.session_state.points)
    canvas_points = _scale_points(canonical_points, CANVAS_SIZE / CANON_SIZE)
    canvas_bg_rgb = cv2.resize(
        bg_rgb,
        (CANVAS_SIZE, CANVAS_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    initial = _points_to_initial_drawing(
        [(point["x"], point["y"]) for point in canvas_points],
        r=8,
        background_rgb=canvas_bg_rgb,
    )

    canvas = st_canvas(
        fill_color="rgba(180, 0, 255, 0.22)",
        stroke_width=3,
        stroke_color="rgba(180, 0, 255, 0.95)",
        # Embed the aligned target directly in Fabric's drawing JSON. The
        # component's background_image helper builds a media URL that is not
        # reliable behind Streamlit Cloud's nested /~/+/ proxy.
        background_color="#000000",
        update_streamlit=True,
        height=CANVAS_SIZE,
        width=CANVAS_SIZE,
        drawing_mode="point" if len(canonical_points) < need else "transform",
        initial_drawing=initial,
        display_toolbar=False,
        point_display_radius=8,
        key=(
            f"canvas_confirm-{st.session_state.cv_cache_key}-"
            f"{st.session_state.canvas_revision}"
        ),
    )

    points = _update_points_from_canvas(canonical_points, canvas.json_data)
    st.session_state.points = points
    st.write(t("marked", lang).format(count=len(points), need=need))

    with st.expander(t("end_self_report", lang), expanded=False):
        st.caption(t("end_self_report_help", lang))
        current_report = st.session_state.end_self_report
        if current_report not in SELF_REPORT_OPTIONS:
            current_report = "none"
        st.session_state.end_self_report = st.selectbox(
            t("end_self_report", lang),
            options=SELF_REPORT_OPTIONS,
            index=SELF_REPORT_OPTIONS.index(current_report),
            format_func=lambda key: t(f"self_{key}", lang),
            label_visibility="collapsed",
        )

    col1, col2 = st.columns([1, 1])
    with col1:
        analyze_clicked = st.button(
            t("analyze", lang),
            use_container_width=True,
            disabled=len(points) < need,
        )
    with col2:
        save_clicked = st.button(t("save_log", lang), use_container_width=True)

    # -------------------------
    # Analyze
    # -------------------------
    if analyze_clicked:
        if len(points) < need:
            st.warning(t("need_points", lang))
            return

        try:
            pts_xy = [(p["x"], p["y"]) for p in points[:need]]

            rect_bgr = st.session_state._rect_photo_bgr
            M_rect_to_canon = st.session_state._M_rect_to_canon
            rect_pts = _canon_to_rect_points(pts_xy, M_rect_to_canon)

            hsvs = []
            for rp in rect_pts:
                try:
                    hsv = sample_contact_color_hsv(rect_bgr, rp, roi_radius=18)
                except TypeError:
                    hsv = sample_contact_color_hsv(rect_bgr, rp[0], rp[1], r=10)
                hsvs.append(hsv)

            scoring = score_hits_color_aware(center, outer, pts_xy, contact_hsvs=hsvs)

            metrics = compute_metrics(points[:need], center=center, outer_radius_px=outer)
            shape = classify_shape(metrics)

            base_advice = next_end_advice(
                metrics,
                shape,
                st.session_state.handedness,
                lang=lang,
                quality=quality,
            )

            cfg = CoachConfig(
                pdf_path=st.session_state.coach_pdf_path,
                mode=st.session_state.coach_mode,
                gguf_path=st.session_state.coach_gguf_path,
                router=st.session_state.coach_router,
            )
            coach = CoachRAG(cfg)
            try:
                advice = coach.enhance_advice(
                    base_advice=base_advice,
                    metrics=metrics,
                    shape=shape,
                    handedness=st.session_state.handedness,
                    lang=lang,
                    scoring=scoring,
                    user_profile=st.session_state.user_profile,
                    log=st.session_state.log,
                    quality=quality,
                    self_report=st.session_state.end_self_report,
                    session_context={
                        "distance_m": int(st.session_state.distance_m),
                        "target_face": st.session_state.target_face,
                    },
                )
            except Exception as exc:
                advice = dict(base_advice)
                advice["rag_error"] = str(exc)

            face_bgr = cv2.cvtColor(bg_rgb, cv2.COLOR_RGB2BGR)
            overlay_hits = _draw_hits_on_face(face_bgr, pts_xy, scoring["scores"])
            overlay_hits_rgb = _bgr_to_rgb_uint8(overlay_hits)

            st.session_state.last_result = {
                "metrics": metrics,
                "shape": shape,
                "advice": advice,
                "scoring": scoring,
                "overlay_hits_rgb": overlay_hits_rgb,
                "quality": quality,
                "color_debug": scoring.get("details", []),
            }
        except Exception:
            logger.exception("Hit analysis failed")
            st.error(t("analysis_error", lang))
            st.caption(t("analysis_error_hint", lang))
            return

        st.rerun()

    # -------------------------
    # Result UI
    # -------------------------
    if st.session_state.last_result:
        res = st.session_state.last_result
        metrics = res["metrics"]
        scoring = res["scoring"]
        advice = res["advice"]
        offset = metrics.get("offset", {}) or {}

        st.divider()
        st.subheader(t("result_overlay", lang))
        image_col, _ = st.columns([7, 4])
        with image_col:
            st.image(res["overlay_hits_rgb"], use_container_width=True)

        st.subheader(t("result_score", lang))
        score_col1, score_col2 = st.columns(2)
        score_col1.metric(t("metric_total", lang), f"{scoring['total']} / {need * 10}")
        score_col2.metric(t("metric_average", lang), f"{scoring['avg']:.2f}")
        st.caption(f"{t('metric_per_arrow', lang)}: {scoring['scores']}")

        st.subheader(t("result_metrics", lang))
        spread_pct = float(metrics.get("spread_ratio") or 0.0) * 100
        offset_pct = float(metrics.get("offset_ratio") or 0.0) * 100
        outlier = metrics.get("outlier", {}) or {}
        shape_label = t(f"shape_{res['shape']}", lang)
        st.write(f"- {t('metric_shape', lang)}: **{shape_label}**")
        st.write(f"- {t('metric_spread', lang)}: **{spread_pct:.1f}%**")
        st.write(f"- {t('metric_direction', lang)}: **{metrics['slope_deg']:.0f}°**")
        st.write(f"- {t('metric_offset', lang)}: **{offset_pct:.1f}%**")
        if outlier.get("present"):
            arrow_label = t("metric_arrow_number", lang).format(number=int(outlier["index"]) + 1)
            core_spread_pct = spread_pct * float(outlier.get("core_spread", 0.0) or 0.0) / max(float(metrics.get("spread", 0.0) or 0.0), 1e-6)
            st.warning(f"{t('metric_outlier', lang)}: **{arrow_label}** · {t('metric_core_spread', lang)} {core_spread_pct:.1f}%")

        st.subheader(t("coach_next", lang))
        st.markdown(f"**{advice.get('title', '')}**")

        single_cue = advice.get("single_cue", advice.get("cue", ""))
        pass_fail = advice.get("pass_fail", "")
        fallback = advice.get("fallback", "")
        mental = advice.get("mental_phrase", "")
        script = advice.get("script", "")

        diagnosis = advice.get("diagnosis", {}) or {}
        feedback = advice.get("feedback", {}) or {}
        if diagnosis:
            confidence = diagnosis.get("confidence_label", diagnosis.get("confidence", ""))
            confidence_reason = diagnosis.get("confidence_reason", "")
            st.info(
                f"**{t('coach_evidence', lang)}**: {diagnosis.get('evidence', '')}\n\n"
                f"**{t('coach_confidence', lang)}**: {confidence} — {confidence_reason}"
            )

        if single_cue:
            st.markdown(f"**{t('coach_one_cue', lang)}**: {single_cue}")
        success = feedback.get("success_criterion", "")
        if success:
            st.success(f"**{t('coach_success', lang)}**: {success}")

        drill = advice.get("drill", {}) or {}
        if isinstance(drill, dict) and drill:
            dur = drill.get("duration_s", None)
            name = drill.get("name", "")
            how = drill.get("how", "")
            if name:
                st.markdown(f"**{t('coach_drill', lang)}**: {name}" + (f" ({dur}s)" if dur is not None else ""))
            if how:
                st.markdown(how)

        if fallback:
            st.markdown(f"**{t('coach_fallback', lang)}**: {fallback}")

        with st.expander(t("coach_reasoning", lang), expanded=False):
            previous_experiment = feedback.get("previous_experiment", {}) or {}
            if previous_experiment:
                st.markdown(
                    f"**{t('coach_previous_test', lang)}**: "
                    f"{previous_experiment.get('verdict', '')} — {previous_experiment.get('title', '')}"
                )
                st.caption(previous_experiment.get("detail", ""))
            why = advice.get("why", "")
            if why:
                st.markdown(f"**{t('coach_why', lang)}**: {why}")
            if diagnosis.get("self_report") not in {None, "none"}:
                st.markdown(f"**{t('coach_self_report', lang)}**: {diagnosis.get('self_report_label', '')}")
            alternatives = feedback.get("alternative_hypotheses", []) or []
            if alternatives:
                st.markdown(f"**{t('coach_alternatives', lang)}**")
                for alternative in alternatives:
                    st.write(f"- {alternative}")
            do_not_change = feedback.get("do_not_change", "")
            if do_not_change:
                st.markdown(f"**{t('coach_do_not_change', lang)}**: {do_not_change}")
            if pass_fail:
                st.markdown(f"**{t('coach_pass_fail', lang)}**: {pass_fail}")
            if diagnosis:
                st.caption(f"{t('coach_trend', lang)}: {diagnosis.get('trend', '')}")
                st.caption(f"{t('coach_handedness', lang)}: {diagnosis.get('handedness_context', '')}")
            if mental:
                st.markdown(f"**{t('coach_mental', lang)}**: {mental}")

        if script:
            with st.expander(t("coach_script", lang), expanded=False):
                st.code(script, language="text")

        source = advice.get("book_source", {}) or {}
        if source:
            with st.expander(t("coach_source", lang), expanded=False):
                st.markdown(f"**{source.get('title', '')}**")
                st.write(source.get("chapter", ""))
                st.write(f"{t('coach_pdf_pages', lang)}: {source.get('pdf_pages', '')}")
                st.caption(source.get("summary", ""))

        with st.expander(t("coach_details", lang)):
            if "rag_error" in advice:
                st.warning(advice["rag_error"])
            st.write({
                "shape": res["shape"],
                "spread_px": metrics.get("spread"),
                "offset_px": offset,
                "diagnosis": diagnosis,
                "source": source,
                "scoring_details": res.get("color_debug", []),
            })

    # -------------------------
    # Save log
    # -------------------------
    if save_clicked:
        if not st.session_state.last_result:
            st.warning(t("analyze_first", lang))
            return

        entry = make_log_entry(
            distance_m=int(st.session_state.distance_m),
            arrows_per_end=int(st.session_state.arrows_per_end),
            handedness=st.session_state.handedness,
            target_face=st.session_state.target_face,
            metrics=st.session_state.last_result["metrics"],
            scoring=st.session_state.last_result["scoring"],
            advice=st.session_state.last_result["advice"],
        )
        st.session_state.log.append(entry)
        st.success(t("saved", lang))

    if st.session_state.log:
        st.divider()
        st.subheader(t("log", lang))
        json_text = export_log_json(st.session_state.log)
        st.download_button(
            label=t("download_log", lang),
            data=json_text.encode("utf-8"),
            file_name="log.json",
            mime="application/json",
            use_container_width=True,
        )
