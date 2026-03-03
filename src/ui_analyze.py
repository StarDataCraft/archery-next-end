# src/ui_analyze.py
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas

from .i18n import t
from .state import goto_step, reset_shot, reset_cv_cache
from .metrics import compute_metrics, classify_shape
from .rules import next_end_advice
from .storage import make_log_entry, export_log_json
from .cv_target import rectify_target, transform_points, propose_hit_points
from .refine_points import refine_points_and_colors, sample_contact_color_hsv
from .scoring import score_hits_color_aware
from .target_face import render_target_face_bgr

CANON_SIZE = 900
CANON_CENTER = (CANON_SIZE / 2.0, CANON_SIZE / 2.0)
CANON_OUTER = CANON_SIZE * 0.45  # 405 px


def _bgr_to_rgb_uint8(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)


def _points_to_initial_drawing(points, r=10):
    objects = []
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
            pts.append({"x": left + r, "y": top + r})
    return pts


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
    """
    Convert canonical points back to rect points using inverse affine.
    """
    if not points_xy:
        return []
    if M_rect_to_canon is None:
        raise TypeError("M_rect_to_canon is None (cannot invert affine transform).")

    if not isinstance(M_rect_to_canon, np.ndarray):
        try:
            M_rect_to_canon = np.array(M_rect_to_canon, dtype=np.float32)
        except Exception as e:
            raise TypeError(f"M_rect_to_canon cannot be converted to np.ndarray: {type(M_rect_to_canon)}") from e

    if M_rect_to_canon.shape != (2, 3):
        raise TypeError(f"M_rect_to_canon must be shape (2,3), got {M_rect_to_canon.shape}")

    Minv = cv2.invertAffineTransform(M_rect_to_canon.astype(np.float32))
    pts = np.array(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.transform(pts, Minv).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in out]


def render_analyze_step():
    lang = st.session_state.language
    st.title(t("title", lang))

    top1, top2, top3 = st.columns([1, 1, 1.4])
    with top1:
        st.session_state.distance_m = st.number_input(
            t("distance", lang),
            min_value=3,
            max_value=90,
            value=int(st.session_state.distance_m),
            step=1,
        )
    with top2:
        st.session_state.arrows_per_end = st.number_input(
            t("arrows", lang),
            min_value=3,
            max_value=12,
            value=int(st.session_state.arrows_per_end),
            step=1,
        )
    with top3:
        face = st.selectbox(
            t("target_face", lang),
            options=["80cm_10ring", "40cm_10ring"],
            index=0 if st.session_state.target_face == "80cm_10ring" else 1,
            format_func=lambda x: t("target_80", lang) if x == "80cm_10ring" else t("target_40", lang),
        )
        if face != st.session_state.target_face:
            st.session_state.target_face = face
            st.session_state.last_result = None
            st.session_state.overlay_image_rgb = None

    colA, colB = st.columns([1, 1])
    with colA:
        if st.button(t("back", lang), use_container_width=True):
            goto_step("handedness")
            st.rerun()
    with colB:
        if st.button(t("clear", lang), use_container_width=True):
            reset_shot()
            reset_cv_cache()
            st.rerun()

    st.divider()
    st.subheader(t("upload", lang))
    file = st.file_uploader("", type=["png", "jpg", "jpeg"])
    if not file:
        st.info("Upload → rectify → propose → refine → map → confirm → analyze")
        return

    img_pil = Image.open(file).convert("RGB")
    img_rgb = np.array(img_pil, dtype=np.uint8)
    cache_key = f"{getattr(file, 'name', 'upload')}-{img_rgb.shape[0]}x{img_rgb.shape[1]}"
    need = int(st.session_state.arrows_per_end)

    if st.session_state.get("cv_cache_key") != cache_key:
        rect_res = rectify_target(img_rgb, out_size=CANON_SIZE)

        face_bgr = render_target_face_bgr(
            st.session_state.target_face,
            size=CANON_SIZE,
            center=CANON_CENTER,
            outer_radius=CANON_OUTER,
            draw_ring_lines=True,
            ring_line_thickness=2,
        )

        coarse = propose_hit_points(
            rect_res.rect_bgr,
            rect_res.center_final,
            rect_res.arrow_present,
            max_points=max(need, 12),
        )

        refined_rect_pts, _ = refine_points_and_colors(
            rect_res.rect_bgr,
            target_center=rect_res.center_final,
            coarse_points=coarse,
            arrow_present=rect_res.arrow_present,
            roi_radius=70,
        )

        auto_pts_canon = transform_points(refined_rect_pts, rect_res.M_rect_to_canon)

        st.session_state.cv_cache_key = cache_key
        st.session_state.overlay_image_rgb = _bgr_to_rgb_uint8(face_bgr)
        st.session_state.auto_points = auto_pts_canon
        st.session_state.points = []
        st.session_state.last_result = None

        st.session_state._geom_center = CANON_CENTER
        st.session_state._geom_outer = CANON_OUTER
        st.session_state._rect_photo_bgr = rect_res.rect_bgr.copy()
        st.session_state._M_rect_to_canon = rect_res.M_rect_to_canon.copy()
        st.session_state.warp_debug = rect_res.debug
        st.session_state.cv_quality = {
            "score": float(rect_res.quality_score),
            "flags": list(rect_res.quality_flags),
        }

    bg_rgb = st.session_state.overlay_image_rgb
    auto_pts = st.session_state.auto_points
    center = st.session_state._geom_center
    outer = st.session_state._geom_outer
    quality = st.session_state.get("cv_quality", None)

    st.subheader(t("tap_points", lang))
    with st.expander("CV debug"):
        st.write(st.session_state.warp_debug)
        if quality is not None:
            st.write({"quality": quality})

    initial = None
    if not st.session_state.points:
        initial = _points_to_initial_drawing(auto_pts[:need], r=10)

    canvas = st_canvas(
        fill_color="rgba(180, 0, 255, 0.22)",
        stroke_width=3,
        stroke_color="rgba(180, 0, 255, 0.95)",
        background_image=Image.fromarray(bg_rgb),
        update_streamlit=True,
        height=700,
        width=700,
        drawing_mode="circle",
        initial_drawing=initial,
        key="canvas_confirm",
    )

    points = _extract_points_from_canvas(canvas.json_data)
    st.session_state.points = points
    st.write(f"Marked: **{len(points)}** / {need}")

    col1, col2 = st.columns([1, 1])
    with col1:
        analyze_clicked = st.button(t("analyze", lang), use_container_width=True)
    with col2:
        save_clicked = st.button(t("save_log", lang), use_container_width=True)

    if analyze_clicked:
        try:
            if len(points) < need:
                st.warning(t("need_points", lang))
                return

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

            advice = next_end_advice(
                metrics,
                shape,
                st.session_state.handedness,
                lang=lang,
                quality=quality,
            )

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
            st.rerun()

        except Exception as e:
            st.error("Analyze crashed.")
            st.exception(e)
            return

    if st.session_state.last_result:
        res = st.session_state.last_result
        metrics = res["metrics"]
        scoring = res["scoring"]
        advice = res["advice"]
        quality = res.get("quality", None)
        offset = metrics.get("offset", {}) or {}

        st.divider()
        st.subheader("Overlay (standard face + hits + scores)")
        st.image(res["overlay_hits_rgb"], use_column_width=False)

        st.subheader("Score (color-aware)")
        st.write(f"- Total: **{scoring['total']}** / {need * 10}")
        st.write(f"- Avg: **{scoring['avg']:.2f}**")
        st.write(f"- Per arrow: {scoring['scores']}")

        with st.expander("Per-arrow debug (radial vs color band)"):
            st.write(res.get("color_debug", []))

        st.subheader("Grouping metrics (reference)")
        st.write(f"- Shape: **{res['shape']}**")
        st.write(f"- Spread (avg): **{metrics['spread']:.1f} px**")
        st.write(f"- Direction: **{metrics['slope_deg']:.0f}°**")
        st.write(f"- Offset dx/dy: **{offset.get('dx', 0.0):.1f} / {offset.get('dy', 0.0):.1f} px**")
        if quality is not None:
            st.write(f"- CV quality: **{float(quality.get('score', 1.0)):.2f}** ({quality.get('flags', [])})")

        # -----------------------------
        # NEW: repetition-first coaching block
        # -----------------------------
        st.subheader("Next end coaching (repetition-first)")
        st.markdown(f"**{advice.get('title','')}**")

        # 1) One cue
        st.markdown(f"**One cue**: {advice.get('single_cue', advice.get('cue',''))}")

        # 2) PASS/FAIL
        st.markdown(f"**PASS/FAIL**: {advice.get('pass_fail','')}")

        # 3) Fallback
        st.markdown(f"**If it breaks**: {advice.get('fallback','')}")

        # 4) Immediate drill
        drill = advice.get("drill", {}) or {}
        if drill:
            st.markdown(f"**Immediate drill ({drill.get('duration_s','?')}s)**: {drill.get('name','')}")
            st.markdown(drill.get("how", ""))

        # 5) Mental phrase
        st.markdown(f"**Mental phrase**: {advice.get('mental_phrase','')}")

        # 6) Script (compact)
        with st.expander("Shot Script (compact)"):
            st.code(advice.get("script", ""), language="text")
            st.write({"stage": advice.get("stage"), "tag": advice.get("tag"), "principle": advice.get("principle")})

    if save_clicked:
        if not st.session_state.last_result:
            st.warning("Analyze first.")
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
        st.success("Saved.")

    if st.session_state.log:
        st.divider()
        st.subheader("Log")
        json_text = export_log_json(st.session_state.log)
        st.download_button(
            label="Download log.json",
            data=json_text.encode("utf-8"),
            file_name="log.json",
            mime="application/json",
            use_container_width=True,
        )
