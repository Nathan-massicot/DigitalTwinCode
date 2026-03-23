import streamlit as st

from data.mock_patients import FEATURE_DEFS

SLIDER_HEIGHT = 180

# -- Components v2 vertical slider --
_vertical_slider = st.components.v2.component(
    "vertical_slider",
    html="""
    <div class="vs-container">
      <div class="vs-value" id="valLabel"></div>
      <input type="range" id="slider" orient="vertical" />
      <div class="vs-cat" id="catLabel"></div>
    </div>
    """,
    css="""
    .vs-container {
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 0;
      margin: 0;
    }
    .vs-value {
      font-size: 13px;
      font-weight: 600;
      color: #ef4444;
      margin-bottom: 6px;
      text-align: center;
      min-height: 18px;
    }
    .vs-cat {
      font-size: 11px;
      color: #475569;
      font-weight: 500;
      margin-top: 4px;
      text-align: center;
      min-height: 14px;
    }
    input[type="range"] {
      writing-mode: vertical-lr;
      direction: rtl;
      -webkit-appearance: slider-vertical;
      appearance: slider-vertical;
      width: 24px;
      cursor: pointer;
      accent-color: #ef4444;
    }
    """,
    js="""
    export default function({ parentElement, setStateValue, data }) {
      const slider = parentElement.querySelector("#slider");
      const valLabel = parentElement.querySelector("#valLabel");
      const catLabel = parentElement.querySelector("#catLabel");

      const minVal = data.min_value;
      const maxVal = data.max_value;
      const step = data.step || 1;
      const height = data.height || 180;
      const options = data.options || [];
      const defaultVal = data.default_value;

      slider.min = minVal;
      slider.max = maxVal;
      slider.step = step;
      slider.style.height = height + "px";

      // Use current state or default
      const currentVal = data._current_value !== undefined ? data._current_value : defaultVal;
      slider.value = currentVal;

      function updateDisplay(val) {
        valLabel.textContent = val;
        if (options.length > 0 && options[val] !== undefined) {
          catLabel.textContent = options[val];
        } else {
          catLabel.textContent = "";
        }
      }

      updateDisplay(parseInt(slider.value));

      slider.oninput = function() {
        const val = parseInt(slider.value);
        updateDisplay(val);
        setStateValue("value", val);
      };
    }
    """,
)

LABEL_CSS = """
<style>
.vs-label {
    transform: rotate(-45deg);
    transform-origin: left bottom;
    white-space: nowrap;
    font-size: 0.72rem;
    font-weight: 600;
    color: #334155;
    text-align: left;
    margin-left: 25px;
    padding-top: 50px;
    margin-bottom: -5px;
}
</style>
"""


def render_sliders() -> dict:
    """Render 20 vertical parameter sliders (10 per row) and return current values."""
    st.markdown(LABEL_CSS, unsafe_allow_html=True)

    values = {}
    cols_per_row = 10
    rows = [FEATURE_DEFS[i:i + cols_per_row] for i in range(0, len(FEATURE_DEFS), cols_per_row)]

    for row_defs in rows:
        cols = st.columns(len(row_defs), gap="small")
        for col, feat in zip(cols, row_defs):
            with col:
                st.markdown(
                    f'<div class="vs-label">{feat["name"]}</div>',
                    unsafe_allow_html=True,
                )

                options = feat.get("options", []) if feat["type"] == "categorical" else []

                key = f"param_{feat['name']}"

                result = _vertical_slider(
                    default={"value": feat["default"]},
                    data={
                        "min_value": feat["min"],
                        "max_value": feat["max"],
                        "default_value": feat["default"],
                        "step": 1,
                        "height": SLIDER_HEIGHT,
                        "options": options,
                        "_current_value": feat["default"],
                    },
                    on_value_change=lambda: None,
                    key=key,
                )

                val = result.value if result.value is not None else feat["default"]
                val = int(val)

                if feat["type"] == "categorical":
                    idx = max(0, min(val, len(feat["options"]) - 1))
                    values[feat["name"]] = feat["options"][idx]
                else:
                    values[feat["name"]] = val

    return values
