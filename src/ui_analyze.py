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
from .cv_target import rectify_target, propose_hit_points
from .scoring import score_hits
from .target_face import render_target_face_bgr
from .refine_points import refine_points


CANON_SIZE = 900
CANON_CENTER = (CANON_SIZE / 2.0, CANON_SIZE / 2.0)
CANON_OUTER = CANON_SIZE * 0.45  # 405 px


def _bgr_to_rgb_uint8(bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.uint8)


def _points_to_initial_drawing(points, r=10):
    objects = []
    for (x, y) in points:
        objects.append({
            "type": "circle",
            "left": float(x - r),
            "top": float(y - r),
            "radius": float(r),
            "fill": "rgba(255, 0, 0, 0.25)",
            "stroke": "rgba(255, 0, 0, 0.9)",
            "strokeWidth": 2,
        })
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


def _map_points_to_canon(points_xy, src_center, src_outer):
    """
    Map points from rectified-photo coords -> canonical face coords.
    x' = Cc + (x - Cs) * (Rc / Rs)
    """
    if src_outer <= 1e-6:
        return points_xy
    sx, sy = src_center
    cx, cy = CANON_CENTER
    scale = CANON_OUTER / float(src_outer)
    out = []
    for (x, y) in points_xy:
        out.append((cx + (x - sx) * scale, cy + (y - sy) * scale))
    return out


def _draw_hits_on_face(face_bgr, points_xy, scores):
    """Draw hit points + scores on the clean face (no color tint)."""
    img = face_bgr.copy()
    PURPLE = (255, 0, 255)  # BGR

    for i, (x, y) in enumerate(points_xy):
        px, py = int(round(x)), int(round(y))
        cv2.circle(img, (px, py), 10, PURPLE, 2)   # purple ring
        cv2.circle(img, (px, py), 2, PURPLE, -1)   # purple dot

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


def render_analyze_step():
    lang = st.session_state.language
    st.title(t("title", lang))

    top1, top2, top3 = st.columns([1, 1, 1.4])
    with top1:
        st.session_state.distance_m = st.number_input(
            t("distance", lang), min_value=3, max_value=90,
            value=int(st.session_state.distance_m), step=1
        )
    with top2:
        st.session_state.arrows_per_end = st.number_input(
            t("arrows", lang), min_value=3, max_value=12,
            value=int(st.session_state.arrows_per_end), step=1
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
        st.info("Upload → rectify → auto candidates → confirm points → analyze (score + cue)")
        return

    img_pil = Image.open(file).convert("RGB")
    img_rgb = np.array(img_pil, dtype=np.uint8)

    cache_key = f"{getattr(file, 'name', 'upload')}-{img_rgb.shape[0]}x{img_rgb.shape[1]}"
    need = int(st.session_state.arrows_per_end)

    if st.session_state.get("cv_cache_key") != cache_key:
        rect_res = rectify_target(img_rgb, out_size=CANON_SIZE)

        # Clean standard face background uses CANON geometry (NOT CV geometry)
        face_bgr = render_target_face_bgr(
            st.session_state.target_face,
            size=CANON_SIZE,
            center=CANON_CENTER,
            outer_radius=CANON_OUTER,
            draw_ring_lines=True,
            ring_line_thickness=2,
        )

        # Propose points from rectified photo
        auto_pts_src = propose_hit_points(
            rect_res.rect_bgr,
            rect_res.center,
            rect_res.arrow_present,
            max_points=12
        )

        # NEW: automatic refinement (no user help)
        auto_pts_refined = refine_points(
            rect_res.rect_bgr,
            rect_res.center,
            auto_pts_src,
            arrow_present=rect_res.arrow_present,
            roi_radius=45,
        )

        # map refined points into canonical
        auto_pts = _map_points_to_canon(auto_pts_refined, rect_res.center, rect_res.outer_radius)

        st.session_state.cv_cache_key = cache_key
        st.session_state.warp_debug = {
            **rect_res.debug,
            "canon_center": CANON_CENTER,
            "canon_outer": CANON_OUTER,
            "auto_pts_src": auto_pts_src,
            "auto_pts_refined": auto_pts_refined,
        }

        st.session_state.overlay_image_rgb = _bgr_to_rgb_uint8(face_bgr)
        st.session_state.auto_points = auto_pts
        st.session_state.points = []
        st.session_state.last_result = None

        # canonical geometry for scoring
        st.session_state._geom_center = CANON_CENTER
        st.session_state._geom_outer = CANON_OUTER

        # keep rectified photo only for debug
        st.session_state._rect_photo_rgb = _bgr_to_rgb_uint8(rect_res.rect_bgr)
        st.session_state._geom_arrow_present = rect_res.arrow_present

    bg_rgb = st.session_state.overlay_image_rgb
    auto_pts = st.session_state.auto_points
    center = st.session_state._geom_center
    outer = st.session_state._geom_outer
    arrow_present = st.session_state._geom_arrow_present

    st.subheader(t("tap_points", lang))
    st.caption("Background is a clean standard target face (correct colors). Ring lines are black. Confirm / add / delete hit points.")

    with st.expander("CV debug"):
        st.write(st.session_state.warp_debug)
        st.write({"arrow_present": arrow_present, "need_points": need})
        st.image(st.session_state._rect_photo_rgb, caption="Rectified photo (debug only)", use_column_width=False)

    initial = None
    if not st.session_state.points:
        initial = _points_to_initial_drawing(auto_pts[:need], r=10)

    canvas = st_canvas(
        fill_color="rgba(255, 0, 0, 0.25)",
        stroke_width=3,
        stroke_color="rgba(255, 0, 0, 0.9)",
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
        if len(points) < need:
            st.warning(t("need_points", lang))
            return

        pts_xy = [(p["x"], p["y"]) for p in points[:need]]

        # scoring uses canonical center/outer -> stable
        scoring = score_hits(center, outer, pts_xy)

        metrics = compute_metrics(points[:need])
        shape = classify_shape(metrics)

        # Advice language follows UI language
        advice = next_end_advice(metrics, shape, st.session_state.handedness, lang=lang)

        face_bgr = cv2.cvtColor(bg_rgb, cv2.COLOR_RGB2BGR)
        overlay_hits = _draw_hits_on_face(face_bgr, pts_xy, scoring["scores"])
        overlay_hits_rgb = _bgr_to_rgb_uint8(overlay_hits)

        st.session_state.last_result = {
            "metrics": metrics,
            "shape": shape,
            "advice": advice,
            "scoring": scoring,
            "overlay_hits_rgb": overlay_hits_rgb,
        }
        st.rerun()

    if st.session_state.last_result:
        res = st.session_state.last_result
        advice = res["advice"]
        metrics = res["metrics"]
        scoring = res["scoring"]

        st.divider()
        st.subheader("Overlay (standard face + hits + scores)")
        st.image(res["overlay_hits_rgb"], use_column_width=False)

        st.subheader("Score")
        st.write(f"- Total: **{scoring['total']}** / {need * 10}")
        st.write(f"- Avg: **{scoring['avg']:.2f}**")
        st.write(f"- Per arrow: {scoring['scores']}")

        st.subheader("Grouping metrics (reference)")
        st.write(f"- Shape: **{res['shape']}**")
        st.write(f"- Spread (avg): **{metrics['spread']:.1f} px**")
        st.write(f"- sx / sy: **{metrics['sx']:.1f} / {metrics['sy']:.1f}**")
        st.write(f"- Direction: **{metrics['slope_deg']:.0f}°**")

        st.subheader("Next end cue")
        st.markdown(f"**{advice['title']}**")
        st.markdown(f"👉 {advice['cue']}")
        with st.expander("Why"):
            st.write(advice["why"])

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
