import * as THREE from "three";

export type RiggedAssetInfo = {
  meshes: number;
  skinnedMeshes: number;
  bones: number;
  vertices: number;
};

export type NeuralRigMap = {
  all: THREE.Bone[];
  armLeft?: THREE.Bone;
  armRight?: THREE.Bone;
  legLeft?: THREE.Bone;
  legRight?: THREE.Bone;
};

export function disposeObject(root: THREE.Object3D) {
  root.traverse((object) => {
    if (!(object instanceof THREE.Mesh)) return;
    object.geometry.dispose();
    const materials = Array.isArray(object.material) ? object.material : [object.material];
    materials.forEach((material) => {
      if ("map" in material && material.map instanceof THREE.Texture) material.map.dispose();
      if ("alphaMap" in material && material.alphaMap instanceof THREE.Texture && material.alphaMap !== material.map) material.alphaMap.dispose();
      if ("displacementMap" in material && material.displacementMap instanceof THREE.Texture) material.displacementMap.dispose();
      material.dispose();
    });
  });
}

export function prepareNeuralCharacter(source: THREE.Group) {
  const character = new THREE.Group();
  character.name = "wallalive-neural-character";
  source.rotation.y = Math.PI;
  character.add(source);
  character.updateMatrixWorld(true);

  let bounds = new THREE.Box3().setFromObject(source);
  const size = bounds.getSize(new THREE.Vector3());
  const scale = 1.62 / Math.max(0.001, size.x, size.y, size.z);
  source.scale.setScalar(scale);
  character.updateMatrixWorld(true);
  bounds = new THREE.Box3().setFromObject(source);
  const center = bounds.getCenter(new THREE.Vector3());
  source.position.x -= center.x;
  source.position.z -= center.z;
  source.position.y += -0.82 - bounds.min.y;
  character.updateMatrixWorld(true);

  const info: RiggedAssetInfo = { meshes: 0, skinnedMeshes: 0, bones: 0, vertices: 0 };
  const bones: THREE.Bone[] = [];
  source.traverse((object) => {
    if (object instanceof THREE.Mesh) {
      info.meshes += 1;
      info.vertices += object.geometry.getAttribute("position")?.count ?? 0;
      object.castShadow = true;
      object.receiveShadow = true;
    }
    if (object instanceof THREE.SkinnedMesh) info.skinnedMeshes += 1;
    if (object instanceof THREE.Bone) {
      info.bones += 1;
      bones.push(object);
      object.userData.wallaliveBaseQuaternion = object.quaternion.clone();
    }
  });

  const root = bones.find((bone) => !(bone.parent instanceof THREE.Bone)) ?? bones[0];
  const rigMap: NeuralRigMap = { all: bones };
  if (root) {
    const rootPosition = root.getWorldPosition(new THREE.Vector3());
    const boneBounds = new THREE.Box3();
    bones.forEach((bone) => boneBounds.expandByPoint(bone.getWorldPosition(new THREE.Vector3())));
    const boneSize = boneBounds.getSize(new THREE.Vector3());
    const branches = root.children.filter((child): child is THREE.Bone => child instanceof THREE.Bone);
    for (const branch of branches) {
      const branchBones: THREE.Bone[] = [];
      branch.traverse((node) => { if (node instanceof THREE.Bone) branchBones.push(node); });
      const endpoint = branchBones.reduce((furthest, bone) => {
        const position = bone.getWorldPosition(new THREE.Vector3());
        return position.distanceTo(rootPosition) > furthest.distanceTo(rootPosition) ? position : furthest;
      }, branch.getWorldPosition(new THREE.Vector3()));
      const dx = endpoint.x - rootPosition.x;
      const dy = endpoint.y - rootPosition.y;
      if (dy < -boneSize.y * 0.13) {
        if (dx < 0) rigMap.legLeft = branch;
        else rigMap.legRight = branch;
      } else if (Math.abs(dx) > boneSize.x * 0.2) {
        if (dx < 0) rigMap.armLeft = branch;
        else rigMap.armRight = branch;
      }
    }
  }
  character.userData.wallaliveRig = rigMap;
  character.userData.reconstruction = {
    method: "AniGen joint mesh-skeleton-skinning reconstruction",
    assetType: "glTF SkinnedMesh",
    topology: "generated full 3D surface",
    viewableDegrees: 360,
    ...info,
  };
  return { character, info, rigMap };
}
