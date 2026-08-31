export type SurfaceMaterial = 0 | 1 | 2;

export function classifySurfaceMaterial(normalX: number, normalY: number, normalZ: number, positionZ: number): SurfaceMaterial {
  const normalLength = Math.hypot(normalX, normalY, normalZ);
  if (normalLength < 1e-8) return 0;

  const cameraFacing = normalZ / normalLength;
  if (cameraFacing > 0.42 && positionZ >= 0) return 1;
  if (cameraFacing < -0.42 && positionZ <= 0) return 2;
  return 0;
}

export function hasRecognizableArtworkSurface(texturedFrontTriangles: number, totalTriangles: number, hasArtworkTexture: boolean) {
  if (!hasArtworkTexture || totalTriangles <= 0) return false;
  return texturedFrontTriangles / totalTriangles >= 0.25;
}
