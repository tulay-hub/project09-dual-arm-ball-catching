"""Bind friction to existing O6 collision meshes; never add collision geometry."""
from pxr import Usd, UsdPhysics, UsdShade


def bind_original_hand_collisions(stage, body_paths, material):
    bindings = []
    for name, path in body_paths.items():
        if not name.startswith(('lh_', 'rh_')) or 'metacarpals' in name:
            continue
        body = stage.GetPrimAtPath(path)
        # Strong binding on the rigid body also resolves on instance-proxy meshes.
        # Traverse only this body's shapes, excluding nested finger rigid bodies.
        UsdShade.MaterialBindingAPI.Apply(body).Bind(
            material, bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose='physics')
        for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            owner = prim
            while owner and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
                owner = owner.GetParent()
            if owner != body:
                continue
            bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
            if not bound or bound.GetPath() != material.GetPath():
                raise RuntimeError(f'Original hand collision material did not resolve: {prim.GetPath()}')
            bindings.append({'body': name, 'path': str(prim.GetPath()),
                             'material': str(bound.GetPath()), 'geometry': 'original_imported_collision'})
    if not bindings or not any(b['body'] == 'lh_hand_base_link' for b in bindings) or not any(
            b['body'] == 'rh_hand_base_link' for b in bindings):
        raise RuntimeError('Both original palm collision meshes must have rubber friction')
    return bindings
