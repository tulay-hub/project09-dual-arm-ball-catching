"""Local real-time display of rendered policy rollout frames; no policy control."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from threading import Thread,Lock
from io import BytesIO
from pathlib import Path
from html import escape
import json,time
from PIL import Image

class LivePolicyView:
    def __init__(self,port,checkpoint,run):
        self.lock=Lock();self.jpeg=b'';self.sequence=0;self.closed=False;self.status={};self.run=Path(run)
        self.checkpoint=checkpoint;viewer=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_GET(self):
                if self.path.startswith('/stream.mjpg'):
                    self.send_response(200);self.send_header('Content-Type','multipart/x-mixed-replace; boundary=frame');self.send_header('Cache-Control','no-store');self.end_headers();previous=-1
                    try:
                        while not viewer.closed:
                            with viewer.lock:jpeg=viewer.jpeg;sequence=viewer.sequence
                            if not jpeg or sequence==previous:time.sleep(.03);continue
                            self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(jpeg)).encode()+b'\r\n\r\n'+jpeg+b'\r\n');self.wfile.flush();previous=sequence
                    except (BrokenPipeError,ConnectionResetError):pass
                    return
                if self.path.startswith('/status.json'):
                    with viewer.lock:payload=json.dumps(viewer.status,ensure_ascii=False).encode()
                    content_type='application/json; charset=utf-8'
                elif self.path.startswith('/frame.jpg'):
                    with viewer.lock:payload=viewer.jpeg
                    content_type='image/jpeg'
                else:
                    payload=viewer.page().encode();content_type='text/html; charset=utf-8'
                self.send_response(200);self.send_header('Content-Type',content_type);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(payload)));self.end_headers()
                try:self.wfile.write(payload)
                except (BrokenPipeError,ConnectionResetError):pass
        for offset in range(10):
            try:self.server=ThreadingHTTPServer(('127.0.0.1',port+offset),Handler);break
            except OSError:
                if offset==9:raise
        self.server.daemon_threads=True;self.url=f'http://127.0.0.1:{self.server.server_port}/'
        self.thread=Thread(target=self.server.serve_forever,kwargs={'poll_interval':.2},daemon=True);self.thread.start()
        (self.run/'live_view.json').write_text(json.dumps({'url':self.url,'checkpoint':checkpoint,'mode':'deterministic_mean_policy','camera':'independent third-person overview','policy_rgbd_resolution':[192,192],'training_updates':False},ensure_ascii=False,indent=2))
    def page(self):
        return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>项目09 · 当前策略播放</title><style>body{margin:0;background:#101823;color:#e7eef7;font:16px system-ui,sans-serif}main{max-width:1100px;margin:24px auto;padding:0 22px}h1{font-size:24px;margin:0 0 12px}.meta{color:#aebed0;font-size:14px;line-height:1.7}.video{margin-top:20px;background:#060b12;border:1px solid #314155;border-radius:12px;overflow:hidden}img{display:block;width:100%;height:auto;min-height:320px;object-fit:contain}.status{padding:14px 0;color:#c7e1d1}</style><main><h1>项目09 · 当前策略实时播放</h1><div class="meta">权重：'''+escape(self.checkpoint)+'''<br>外部来球 → 掌面捕获 → 收手抓稳 · 策略均值推理 · 自动循环回合</div><div class="video"><img src="/stream.mjpg" alt="仿真实时画面"></div><div class="status" id="status">等待第一帧…</div><div class="meta">画面来自独立观察相机；策略仍使用机身相机的 192×192 RGB-D。</div></main><script>async function tick(){try{let s=await(await fetch('/status.json',{cache:'no-store'})).json();document.querySelector('#status').textContent=`${s.side||''} · ${s.phase||'初始化'}　|　仿真 ${s.simulated_seconds||0} 秒　|　完成 ${s.episodes||0} 回合 · 成功 ${s.successes||0} 次`;}catch(e){document.querySelector('#status').textContent='播放连接已结束';}}setInterval(tick,800);tick();</script></html>'''
    def publish(self,rgb,status):
        output=BytesIO();Image.fromarray(rgb).save(output,format='JPEG',quality=88)
        with self.lock:self.jpeg=output.getvalue();self.sequence+=1;self.status=dict(status,frame=self.sequence)
        if self.sequence%30==0:(self.run/'live_status.json').write_text(json.dumps(self.status,ensure_ascii=False,indent=2))
    def close(self):
        self.closed=True;self.server.shutdown();self.server.server_close()
