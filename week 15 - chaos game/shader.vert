// Passthrough vertex shader

precision mediump float;

attribute vec3 aPosition;
attribute vec2 aTexCoord;
varying vec2 vTexCoord;
uniform mat4 uModelViewProjectionMatrix;

void main() {
  vTexCoord = aTexCoord;
  gl_Position = uModelViewProjectionMatrix * vec4(aPosition, 1.0);
}
