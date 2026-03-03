import streamlit as st
from PIL import Image
import numpy as np
import cv2

from streamlit_drawable_canvas import st_canvas

from .i18n import t
from .state import goto_step, reset_shot
from .metrics import compute_metrics, classify_shape
from .rules import next_end_advice
from .storage import make_log_entry, export_log_json
from .cv_target import detect_target_and_warp, propose_hit_points


def _bgr_to_rgb_uint8(bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.uint8)


def _points_to_initial_drawing(points, r=10):
    """
    Convert list of (x,y) into drawable-canvas initial drawing json.
    Using circles so user can move/delete easily.
    """
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

    top1, top2, top3 = st.columns([1, 1, 2])
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

    colA, colB = st.columns([1, 1])
    with colA:
        if st.button(t("back", lang), use_container_width=True):
            goto_step("handedness")
            st.rerun()
    with colB:
        if st.button(t("clear", lang), use_container_width=True):
            reset_shot()
            # also clear auto cache for this image
            st.session_state.pop("auto_points", None)
            st.session_state.pop("warped_image_rgb", None)
            st.session_state.pop("warp_debug", None)
            st.rerun()

    st.divider()
    st.subheader(t("upload", lang))
    file = st.file_uploader("", type=["png", "jpg", "jpeg"])

    if not file:
        st.info("画像を選ぶ → 自動候補点が出る → 不要な点を消す/足す → Analyze")
        return

    # Read image
    img_pil = Image.open(file).convert("RGB")
    img_rgb = np.array(img_pil, dtype=np.uint8)

    # Run warp + auto proposals once per upload (keyed by file name + size)
    cache_key = f"{getattr(file, 'name', 'upload')}-{img_rgb.shape[0]}x{img_rgb.shape[1]}"
    if st.session_state.get("cv_cache_key") != cache_key:
        warp_res = detect_target_and_warp(img_rgb, out_size=900)
        warped_rgb = _bgr_to_rgb_uint8(warp_res.warped_bgr)
        auto_pts = propose_hit_points(warp_res.warped_bgr, warp_res.center, max_points=12)

        st.session_state.cv_cache_key = cache_key
        st.session_state.warped_image_rgb = warped_rgb
        st.session_state.auto_points = auto_pts
        st.session_state.warp_debug = warp_res.debug
        st.session_state.last_result = None
        st.session_state.points = []

    warped_rgb = st.session_state.warped_image_rgb
    auto_pts = st.session_state.auto_points

    st.subheader("Auto candidates (確認して編集)")
    st.caption("赤い点が自動候補。不要な点は消して、足りなければ追加してください（円を描くと点になります）。")

    # show debug briefly
    with st.expander("CV debug"):
        st.write(st.session_state.warp_debug)

    need = int(st.session_state.arrows_per_end)

    # Pre-fill canvas with auto candidates only if user hasn't edited yet
    initial = None
    if not st.session_state.points:
        initial = _points_to_initial_drawing(auto_pts[:need], r=10)

    canvas = st_canvas(
        fill_color="rgba(255, 0, 0, 0.25)",
        stroke_width=3,
        stroke_color="rgba(255, 0, 0, 0.9)",
        background_image=Image.fromarray(warped_rgb),
        update_streamlit=True,
        height=700,
        width=700,
        drawing_mode="circle",
        initial_drawing=initial,
        key="canvas_auto",
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

        metrics = compute_metrics(points[:need])
        shape = classify_shape(metrics)
        advice = next_end_advice(metrics, shape, st.session_state.handedness)

        st.session_state.last_result = {
            "metrics": metrics,
            "shape": shape,
            "advice": advice
        }
        st.rerun()

    if st.session_state.last_result:
        res = st.session_state.last_result
        advice = res["advice"]
        metrics = res["metrics"]

        st.divider()
        st.subheader("結果")
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
            metrics=st.session_state.last_result["metrics"],
            advice=st.session_state.last_result["advice"],
        )
        st.session_state.log.append(entry)
        st.success("保存しました。")

    if st.session_state.log:
        st.divider()
        st.subheader("ログ")
        st.caption("まずは JSON でエクスポート（あとでCSVも足せる）")
        json_text = export_log_json(st.session_state.log)
        st.download_button(
            label="Download log.json",
            data=json_text.encode("utf-8"),
            file_name="log.json",
            mime="application/json",
            use_container_width=True,
        )
