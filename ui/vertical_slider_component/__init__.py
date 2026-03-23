import os
import streamlit.components.v1 as components

_COMPONENT_DIR = os.path.dirname(os.path.abspath(__file__))

_vertical_slider = components.declare_component(
    "vertical_slider",
    path=_COMPONENT_DIR,
)


def vertical_slider(
    min_value: int,
    max_value: int,
    default_value: int,
    step: int = 1,
    height: int = 180,
    options: list | None = None,
    key: str | None = None,
) -> int:
    """Render a vertical slider and return its current value."""
    val = _vertical_slider(
        min_value=min_value,
        max_value=max_value,
        default_value=default_value,
        step=step,
        height=height,
        options=options or [],
        key=key,
        default=default_value,
    )
    return int(val) if val is not None else default_value
