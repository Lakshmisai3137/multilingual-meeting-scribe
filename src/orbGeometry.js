const tau = Math.PI * 2

// A corrugated spherical surface sampled as continuous latitude filaments.
export function surface(latitude, longitude) {
  const c = Math.cos(latitude)
  const s = Math.sin(latitude)
  const wave = .072 * Math.sin(3 * longitude + 4 * latitude)
    + .048 * Math.cos(5 * longitude - 3 * latitude)
    + .031 * Math.sin(7 * longitude + 5 * latitude)
  const radius = 1 + c * wave
  const bent = latitude + .14 * c * Math.sin(longitude * 2.5 + latitude * 2 + .6) * Math.cos(longitude / 2)
  return [
    radius * Math.cos(bent) * Math.cos(longitude),
    radius * Math.sin(bent),
    radius * Math.cos(bent) * Math.sin(longitude),
  ]
}

export function makeFilaments(rows = 84, columns = 190) {
  const lines = []
  for (let row = 1; row < rows; row++) {
    const latitude = (row / rows - .5) * Math.PI
    for (let col = 0; col < columns; col++) {
      lines.push(...surface(latitude, tau * col / columns), ...surface(latitude, tau * (col + 1) / columns))
    }
  }
  return new Float32Array(lines)
}
