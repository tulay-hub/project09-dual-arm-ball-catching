"""Playback-only Kit layout and viewport cost controls; policy camera unchanged."""
import json
from pathlib import Path


def configure_compact_playback(run,physics_hz):
    import omni.ui as ui
    from omni.kit.viewport.utility import get_active_viewport,get_viewport_from_window_name
    hidden=[]
    for window in ui.Workspace.get_windows():
        title=window.title
        if any(word in title.lower() for word in ['stage','property','content','console','timeline',
                                                  'layers','isaac sim','extension','asset browser']):
            window.visible=False;hidden.append(title)
    viewport=get_viewport_from_window_name("Visualizer Viewport") or get_active_viewport()
    if viewport is not None:
        viewport.fill_frame=False
        viewport.set_texture_resolution((960,540))
    report={'hidden_windows':hidden,'viewport_resolution':list(viewport.get_texture_resolution()) if viewport is not None else None,
            'policy_camera_resolution_unchanged':True,'physics_hz':physics_hz,'control_hz':60}
    (Path(run)/'compact_playback.json').write_text(json.dumps(report,indent=2))
    print('[COMPACT_PLAYBACK]',json.dumps(report),flush=True)
    return report
