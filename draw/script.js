canvas = document.getElementById('canvas')
h = parseInt(canvas.getAttribute("height"))
w = parseInt(canvas.getAttribute("width"))
ctx = canvas.getContext('2d')

var zip = JSZip()
last_index = 0

document.getElementById('btn_clear').onclick = clear
document.getElementById('btn_save').onclick = save
document.getElementById('btn_download').onclick = download

function clear() {
  ctx.fillStyle = "#000"
  ctx.fillRect(0, 0, w, h)
}

function draw_circle(x, y) {
  ctx.fillStyle = "#FFF"
  ctx.beginPath()
  ctx.arc(x, y, 10, 0, 2 * Math.PI)
  ctx.fill()
}

function getMousePos(canvas, evt) {
  var rect = canvas.getBoundingClientRect();
  return {
    x: evt.clientX - rect.left,
    y: evt.clientY - rect.top
  };
}

function save() {
  zip.file(last_index + ".png", canvas.toDataURL().replace(/^data:image\/(png|jpg);base64,/, ""), {
    base64: true
  });
  last_index++
  clear()
}

function download() {
  zip.generateAsync({
      type: "blob"
    })
    .then(function(content) {
      saveAs(content, "images.zip");
    })
}
var mouse_down = false
document.body.onmousedown = function() {
  mouse_down = true
}
document.body.onmouseup = function() {
  mouse_down = false
}
canvas.addEventListener('mousemove', function(evt) {
  var mousePos = getMousePos(canvas, evt);
  if (mouse_down) {
    draw_circle(mousePos.x, mousePos.y)
  }
}, false)
clear()
