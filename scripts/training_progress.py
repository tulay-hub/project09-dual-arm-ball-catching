"""Checkpoint progress semantics: saved iteration index versus completed updates."""
def completed_updates_from_checkpoint(saved, filename):
    info=saved.get('infos') or {}
    if 'completed_updates' in info:return int(info['completed_updates'])
    if filename=='model_initial.pt':return 0
    return int(saved['iter'])+1
