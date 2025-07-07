precision mediump float;

varying vec2 vTexCoord;
uniform sampler2D u_texture;
uniform float u_decay;

void main() {
  vec4 color = texture2D(u_texture, vTexCoord);
  vec3 decayed = color.rgb - 0.002;
  gl_FragColor = vec4(max(decayed, 0.0), 1.0);
}
