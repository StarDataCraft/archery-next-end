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
from .cv_target import rectify_target, propose_hit_points, draw_overlay
from .scoring import ring_radii_px, score_hits


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


def render_analyze_step():
    lang = st.session_state.language
    st.title(t("title", lang))

    # --- controls
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
        # target face select
        face = st.selectbox(
            t("target_face", lang),
            options=["80cm_10ring", "40cm_10ring"],
            index=0 if st.session_state.target_face == "80cm_10ring" else 1,
            format_func=lambda x: t("target_80", lang) if x == "80cm_10ring" else t("target_40", lang),
        )
        if face != st.session_state.target_face:
            st.session_state.target_face = face
            st.session_state.last_result = None
            # scoring overlay depends on rings, so clear cached overlay
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
        st.info("写真 → 自動で正対補正＆候補点 → 不要な点を消す/足す → Analyze（スコアも出ます）")
        return

    img_pil = Image.open(file).convert("RGB")
    img_rgb = np.array(img_pil, dtype=np.uint8)

    cache_key = f"{getattr(file, 'name', 'upload')}-{img_rgb.shape[0]}x{img_rgb.shape[1]}"
    need = int(st.session_state.arrows_per_end)

    # --- Run CV only when new upload
    if st.session_state.get("cv_cache_key") != cache_key:
        rect_res = rectify_target(img_rgb, out_size=900)

        # ring geometry
        rings = ring_radii_px(rect_res.outer_radius, st.session_state.target_face)

        # propose points
        auto_pts = propose_hit_points(
            rect_res.rect_bgr,
            rect_res.center,
            rect_res.arrow_present,
            max_points=12
        )

        # overlay (with rings only; hits drawn after user confirms)
        overlay = draw_overlay(
            rect_res.rect_bgr,
            rect_res.center,
            rings,
            points_xy=[],
            scores=[],
        )

        st.session_state.cv_cache_key = cache_key
        st.session_state.warp_debug = rect_res.debug
        st.session_state.warped_image_rgb = _bgr_to_rgb_uint8(rect_res.rect_bgr)
        st.session_state.overlay_image_rgb = _bgr_to_rgb_uint8(overlay)
        st.session_state.auto_points = auto_pts
        st.session_state.points = []
        st.session_state.last_result = None

        # store geometry for scoring
        st.session_state._geom_center = rect_res.center
        st.session_state._geom_outer = rect_res.outer_radius
        st.session_state._geom_rings = rings
        st.session_state._geom_arrow_present = rect_res.arrow_present

    overlay_rgb = st.session_state.overlay_image_rgb
    auto_pts = st.session_state.auto_points
    center = st.session_state._geom_center
    outer = st.session_state._geom_outer
    rings = st.session_state._geom_rings
    arrow_present = st.session_state._geom_arrow_present

    st.subheader(t("tap_points", lang))
    st.caption("黄色い円＝リング（計算基準） / 赤点＝候補。不要なら削除、足りなければ追加。")

    with st.expander("CV debug"):
        st.write(st.session_state.warp_debug)
        st.write({"arrow_present": arrow_present, "need_points": need})

    # Pre-fill canvas with candidates if user hasn't edited yet
    initial = None
    if not st.session_state.points:
        initial = _points_to_initial_drawing(auto_pts[:need], r=10)

    canvas = st_canvas(
        fill_color="rgba(255, 0, 0, 0.25)",
        stroke_width=3,
        stroke_color="rgba(255, 0, 0, 0.9)",
        background_image=Image.fromarray(overlay_rgb),
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

        # Use first N points
        pts_xy = [(p["x"], p["y"]) for p in points[:need]]

        scoring = score_hits(center, outer, pts_xy)

        # metrics for coaching logic
        metrics = compute_metrics(points[:need])
        shape = classify_shape(metrics)
        advice = next_end_advice(metrics, shape, st.session_state.handedness)

        # draw overlay with hits+scores
        rect_bgr = cv2.cvtColor(st.session_state.warped_image_rgb, cv2.COLOR_RGB2BGR)
        overlay2 = draw_overlay(rect_bgr, center, rings, pts_xy, scoring["scores"])
        overlay2_rgb = _bgr_to_rgb_uint8(overlay2)

        st.session_state.last_result = {
            "metrics": metrics,
            "shape": shape,
            "advice": advice,
            "scoring": scoring,
            "overlay_hits_rgb": overlay2_rgb,
        }
        st.rerun()

    if st.session_state.last_result:
        res = st.session_state.last_result
        advice = res["advice"]
        metrics = res["metrics"]
        scoring = res["scoring"]

        st.divider()
        st.subheader("Overlay（リング + 命中 + 点数）")
        st.image(res["overlay_hits_rgb"], use_column_width=False)

        st.subheader("スコア")
        st.write(f"- Total: **{scoring['total']}** / {need * 10}")
        st.write(f"- Avg: **{scoring['avg']:.2f}**")
        st.write(f"- Per arrow: {scoring['scores']}")

        st.subheader("グルーピング指標（参考）")
        st.write(f"- Shape: **{res['shape']}**")
        st.write(f"- Spread (avg): **{metrics['spread']:.1f} px**")
        st.write(f"- sx / sy: **{metrics['sx']:.1f} / {metrics['sy']:.1f}**")
        st.write(f"- Direction: **{metrics['slope_deg']:.0f}°**")

        st.subheader("次のエンド：ワンキュー")
        st.markdown(f"**{advice['title']}**")
        st.markdown(f"👉 {advice['cue']}")
        with st.expander("理由（短く）"):
            st.write(advice["why"])

    if save_clicked:
        if not st.session_state.last_result:
            st.warning("先に分析してください。")
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
        st.success("保存しました。")

    if st.session_state.log:
        st.divider()
        st.subheader("ログ")
        json_text = export_log_json(st.session_state.log)
        st.download_button(
            label="Download log.json",
            data=json_text.encode("utf-8"),
            file_name="log.json",
            mime="application/json",
            use_container_width=True,
        )
