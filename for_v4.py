import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import ast
import math
import re
import os
import plotly.express as px
import plotly.graph_objects as go
import heapq
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
# ---- Orquestador de tareas (Redis) ----
import json
import time
import uuid
try:
    import redis
    REDIS_DISPONIBLE = True
except ImportError:
    REDIS_DISPONIBLE = False

# ---- Entorno de ejecucion: local o Streamlit Cloud ----
def _detectar_nube():
    """True si la app corre en Streamlit Community Cloud (o se fuerza con SIMULADOR_MODO_NUBE=1).
    En la nube no hay acceso a carpetas del usuario ni Redis local: la app usa subida de archivos y descargas."""
    forzado = os.environ.get("SIMULADOR_MODO_NUBE", "").strip().lower()
    if forzado in ("1", "true", "si", "sí"):
        return True
    if forzado in ("0", "false", "no"):
        return False
    try:
        ruta = os.path.abspath(__file__).replace("\\", "/")
        return (ruta.startswith("/mount/src/") or os.environ.get("HOME", "") == "/home/adminuser"
                or bool(os.environ.get("STREAMLIT_SHARING_MODE")))
    except Exception:
        return False


ES_NUBE = _detectar_nube()


def _cfg_nube(nombre, defecto=None):
    """Lee un ajuste desde variable de entorno o, si no existe, desde st.secrets (Settings > Secrets en la nube)."""
    v = os.environ.get(nombre)
    if v not in (None, ""):
        return v
    try:
        if nombre in st.secrets:
            return str(st.secrets[nombre])
    except Exception:
        pass
    return defecto


# ---- Limite de subida de archivos (MWD > 200 MB) ----
# 1) Intento en caliente (puede no aplicar segun la version de Streamlit).
try:
    from streamlit import config as _st_config
    _st_config.set_option("server.maxUploadSize", 600)
except Exception:
    pass
# 2) Persistente: crea .streamlit/config.toml si no existe (aplica al relanzar la app).
try:
    _cfg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".streamlit")
    _cfg_path = os.path.join(_cfg_dir, "config.toml")
    if not ES_NUBE and not os.path.exists(_cfg_path):  # en la nube el config.toml debe ir en el repositorio
        os.makedirs(_cfg_dir, exist_ok=True)
        with open(_cfg_path, "w", encoding="utf-8") as _f:
            _f.write("[server]\nmaxUploadSize = 600\n")
except Exception:
    pass

st.set_page_config(page_title="Simulador de Tuneles", layout="wide")
st.title("Simulador de Avance de Tuneles - Monte Carlo")


# ======================================================================
# #####REGISTRO DE ACTIVIDAD (LOGS) — panel al final de la pagina#####
# ======================================================================
# Guarda en memoria (ultimos 1000 eventos) lo que va pasando: botones pulsados, archivos cargados,
# mensajes st.success/info/warning/error, avisos de Python/Streamlit y progreso de la simulacion.
# Formato: Fecha-hora | Nivel | Mensaje. El panel se dibuja al final del script (ver mas abajo).
# El almacen vive en un modulo falso de sys.modules para sobrevivir a los reruns de Streamlit.
import sys as _sys, types as _types, threading as _threading, collections as _collections
import logging as _logging, warnings as _warnings, html as _html, functools
from datetime import datetime as _datetime

_LOGS = _sys.modules.get('_simulador_logs_store')
if _LOGS is None:
    _LOGS = _types.ModuleType('_simulador_logs_store')
    _LOGS.items = _collections.deque(maxlen=1000)
    _LOGS.lock = _threading.Lock()
    _LOGS.prev = set()
    _LOGS.cur = set()
    _LOGS.ultimo_run = None
    _LOGS.instalado = False
    _LOGS.depth = 0
    _LOGS.menu = None
    _sys.modules['_simulador_logs_store'] = _LOGS


if not hasattr(_LOGS, 'depth'):
    _LOGS.depth = 0
    _LOGS.menu = None


def log_evento(nivel, mensaje, dedupe=False):
    """Agrega una linea al registro. nivel: INFO, SUCCESS, WARNING, ERROR, CRITICAL.
    dedupe=True evita repetir un mensaje que ya se mostro en la corrida anterior (mensajes
    persistentes que Streamlit vuelve a dibujar en cada rerun)."""
    try:
        txt = re.sub(r'\*\*|`', '', str(mensaje))
        txt = re.sub(r'\s+', ' ', txt).strip()
        if len(txt) > 600:
            txt = txt[:600] + '…'
        if not txt:
            return
        nivel = str(nivel).upper()
        menu = getattr(_LOGS, 'menu', None)
        if menu:
            txt = f"[{menu}] {txt}"
        repetido = False
        with _LOGS.lock:
            if dedupe:
                clave = (nivel, txt)
                repetido = clave in _LOGS.prev or clave in _LOGS.cur
                _LOGS.cur.add(clave)
            if not repetido:
                _LOGS.seq = getattr(_LOGS, 'seq', 0) + 1
                _LOGS.items.append((_LOGS.seq, _datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], nivel, txt))
        # Alimenta la ventana de flujo de datos (fuera del lock: las funciones del flujo toman su propio lock).
        _crudo = txt[len(menu) + 3:] if (menu and txt.startswith(f"[{menu}] ")) else txt
        if dedupe:  # mensaje persistente: mientras Streamlit lo siga dibujando, el menu "lo tiene vigente"
            _vig_fn = globals().get('_flujo_msg_vigente')
            if _vig_fn is not None:
                _vig_fn(nivel, _crudo, menu)
        if repetido:
            return
        _flujo_fn = globals().get('_flujo_registrar')
        if _flujo_fn is not None:
            _flujo_fn(nivel, _crudo, menu, bool(dedupe))
        # Tiempo real: los eventos importantes (mensajes, botones, cambios, archivos) redibujan logs y flujo
        # en el momento, sin esperar al final de la corrida (se ignora si no hay contexto de Streamlit).
        _es_acc = globals().get('_flujo_es_accion')
        if nivel in ('SUCCESS', 'WARNING', 'ERROR', 'CRITICAL') or (_es_acc is not None and _es_acc(_crudo)):
            _rf = globals().get('_logs_refrescar')
            if _rf is not None:
                _rf(gap=0.2)
    except Exception:
        pass


class _HandlerPanelLogs(_logging.Handler):
    def emit(self, record):
        try:
            if getattr(record, '_panel_logged', False):  # ya registrado por otro logger padre
                return
            record._panel_logged = True
            detalle = ''
            if record.exc_info and record.exc_info[0] is not None:
                detalle = f" -> {record.exc_info[0].__name__}: {record.exc_info[1]}"
            _LOGS.log(record.levelname, f"[{record.name}] {record.getMessage()}{detalle}")
        except Exception:
            pass


class _HandlerAccesoHTTP(_logging.Handler):
    """Convierte las lineas de acceso del servidor (uvicorn / tornado), que antes solo salian en la
    terminal (GET /media/xxxx.xlsx 200), en eventos del panel con el formato estandar
    Fecha-hora | Nivel | Mensaje. Las descargas (/media/) se registran siempre; el resto de
    rutas solo si fallan (HTTP >= 400)."""
    _TIPOS = {'.xlsx': 'Excel (.xlsx)', '.xls': 'Excel (.xls)', '.csv': 'CSV (.csv)', '.zip': 'ZIP (.zip)',
              '.pdf': 'PDF (.pdf)', '.png': 'Imagen PNG', '.html': 'HTML (.html)', '.json': 'JSON (.json)',
              '.txt': 'Texto (.txt)'}

    def emit(self, record):
        try:
            a = record.args
            if isinstance(a, tuple) and len(a) >= 5:  # uvicorn.access: (cliente, metodo, ruta, http, estado)
                metodo, ruta, estado = str(a[1]), str(a[2]), int(a[4])
            else:
                texto = record.getMessage()
                m = re.search(r'"?(GET|POST|PUT|DELETE|HEAD|PATCH)\s+(\S+)\s+HTTP/[\d.]+"?\s+(\d{3})', texto)
                if m:
                    metodo, ruta, estado = m.group(1), m.group(2), int(m.group(3))
                else:  # tornado.access: "200 GET /ruta (::1) 12.34ms"
                    m = re.search(r'(\d{3})\s+(GET|POST|PUT|DELETE|HEAD|PATCH)\s+(\S+)', texto)
                    if not m:
                        return
                    estado, metodo, ruta = int(m.group(1)), m.group(2), m.group(3)
            ruta = ruta.split('?')[0]
            es_media = ruta.startswith('/media/')
            if not es_media and estado < 400:
                return
            from http import HTTPStatus
            try:
                frase = HTTPStatus(estado).phrase
            except Exception:
                frase = ''
            nivel = 'INFO' if estado < 400 else ('WARNING' if estado < 500 else 'ERROR')
            if es_media:
                base = os.path.basename(ruta)
                ext = os.path.splitext(base)[1].lower()
                tipo = self._TIPOS.get(ext, f"archivo {ext}" if ext else 'archivo')
                if estado < 300:
                    msg = f"Descarga entregada al navegador: {tipo}, id {base[:8]} (HTTP {estado} {frase})"
                else:
                    msg = f"Falló la descarga de {tipo}, id {base[:8]} (HTTP {estado} {frase})"
            else:
                msg = f"Solicitud {metodo} {ruta} respondió HTTP {estado} {frase}"
            _LOGS.log(nivel, msg)
        except Exception:
            pass


def _logs_nuevo_run():
    """Se llama al inicio de cada corrida del script: rota los mensajes 'ya vistos' y se asegura
    de que los loggers de Streamlit/Python escriban tambien en el panel."""
    _LOGS.run_id = getattr(_LOGS, 'run_id', 0) + 1  # id de esta corrida (el flujo lo usa para saber que menus se dibujan ahora)
    try:
        _f_prev = getattr(_LOGS, 'flujo', None)
        if _f_prev is not None:
            _f_prev['fin'] = False
    except Exception:
        pass
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx()
        marca = getattr(ctx, 'cursors', None)
        if marca is not None and marca is not _LOGS.ultimo_run:
            _LOGS.ultimo_run = marca  # se guarda la referencia: asi la identidad no se reutiliza
            _LOGS.prev, _LOGS.cur = _LOGS.cur, set()
    except Exception:
        pass
    try:
        st.session_state['_log_run_n'] = st.session_state.get('_log_run_n', 0) + 1
    except Exception:
        pass
    # Logs de acceso HTTP del servidor (uvicorn en Streamlit nuevo, tornado en el antiguo). No propagan
    # al logger raiz, por eso se les agrega un handler propio; se reengancha en cada corrida.
    h_acc = getattr(_LOGS, 'handler_acceso', None)
    if h_acc is None:
        h_acc = _HandlerAccesoHTTP(level=_logging.INFO)
        _LOGS.handler_acceso = h_acc
    for _n_acc in ('uvicorn.access', 'tornado.access'):
        _lg_acc = _logging.getLogger(_n_acc)
        if h_acc not in _lg_acc.handlers:
            _lg_acc.addHandler(h_acc)
        if _lg_acc.level == _logging.NOTSET or _lg_acc.level > _logging.INFO:
            _lg_acc.setLevel(_logging.INFO)
    h = getattr(_LOGS, 'handler', None)
    if h is None:
        h = _HandlerPanelLogs(level=_logging.INFO)
        _LOGS.handler = h
    raiz = _logging.getLogger()
    if not raiz.handlers:  # conserva la salida por consola que Python daria sin handlers
        raiz.addHandler(_logging.StreamHandler())
    for lg in [raiz] + [lg for n, lg in list(_logging.root.manager.loggerDict.items())
                        if n.startswith('streamlit') and isinstance(lg, _logging.Logger)]:
        if h not in lg.handlers:
            lg.addHandler(h)


# ---- Apoyo al flujo de datos: widgets sin key, graficos/tablas visualizados, widgets de resultados ----
# Los selectores que eligen QUE resultado se ve (viven dentro del menu 16) se atribuyen al menu 17.
_FLUJO_MENU_POR_KEY = {
    'modo_visualizacion_resultados': 'M17 · Avance de los Túneles',
    'sim_seleccionada_global': 'M17 · Avance de los Túneles',
}


def _ocurrencia(base):
    """N° de veces que se ha visto 'base' en ESTA corrida (distingue widgets sin key con el mismo texto)."""
    run_n = st.session_state.get('_log_run_n', 0)
    d = getattr(_LOGS, 'occ', None)
    if d is None or getattr(_LOGS, 'occ_run', None) != run_n:
        d = {}
        _LOGS.occ = d
        _LOGS.occ_run = run_n
    d[base] = d.get(base, 0) + 1
    return d[base]


def _desc_vista(tipo, obj):
    """Devuelve (descripcion, firma, extra) de un grafico o tabla mostrados, o None si no se puede describir."""
    if tipo == 'grafico':
        tit = ''
        lay = getattr(obj, 'layout', None)
        if lay is not None:
            t = getattr(lay, 'title', None)
            tit = (getattr(t, 'text', '') or '') if t is not None else ''
        tit = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', str(tit))).strip()
        trazas = list(getattr(obj, 'data', None) or [])
        firma = [tit, len(trazas)]
        for tr in trazas[:12]:
            y = getattr(tr, 'y', None)
            x = getattr(tr, 'x', None)
            ref = y if y is not None else x
            n = len(ref) if ref is not None else 0
            ult = str(ref[-1])[:12] if n else ''
            firma.append((str(getattr(tr, 'type', '')), str(getattr(tr, 'name', '')), n, ult))
        return (tit[:90] or 'sin título'), tuple(firma), (f"{len(trazas)} series" if trazas else '')
    df = obj.data if isinstance(getattr(obj, 'data', None), pd.DataFrame) else obj
    if not isinstance(df, pd.DataFrame):
        return None
    cols = tuple(map(str, df.columns))
    prim = str(df.iloc[0].tolist())[:60] if len(df) else ''
    ult = str(df.iloc[-1].tolist())[:60] if len(df) else ''
    desc = ', '.join(cols[:4]) + ('…' if len(cols) > 4 else '')
    return (desc or 'sin columnas'), (cols, len(df), prim, ult), f"{len(df)} filas × {len(cols)} columnas"


def _registrar_vista(tipo, obj):
    """Registra 'Visualizó gráfico/tabla' dentro del menu actual cuando aparece, reaparece o cambia su contenido."""
    try:
        if not getattr(_LOGS, 'menu', None):
            return
        info = _desc_vista(tipo, obj)
        if info is None:
            return
        desc, firma, extra = info
        ss = st.session_state
        base = ss.setdefault('_log_vistas', {})
        run_n = ss.get('_log_run_n', 0)
        b = f"{_LOGS.menu}|vista|{tipo}|{desc}"
        ident = f"{b}|#{_ocurrencia(b)}"
        previo = base.get(ident)
        base[ident] = (firma, run_n)
        if previo is not None and previo[1] >= run_n - 1 and previo[0] == firma:
            return
        etq = 'gráfico' if tipo == 'grafico' else 'tabla'
        _LOGS.log('INFO', f"Visualizó {etq} «{desc}»" + (f" ({extra})" if extra else ''))
    except Exception:
        pass


def _logs_instalar_parches():
    """Una sola vez por proceso: envuelve st.success/info/warning/error/exception/toast, los botones
    y los file_uploader para que cada accion quede registrada, y redirige warnings.warn al panel
    sin dejar de imprimirlos en la consola."""
    if getattr(_LOGS, 'instalado', False) == 'v3':
        return
    _LOGS.instalado = 'v3'

    from streamlit.delta_generator import DeltaGenerator as _DG

    def _fmt_valor(v):
        try:
            if isinstance(v, bool):
                return 'sí' if v else 'no'
            if isinstance(v, float):
                return f"{v:.6g}"
            if isinstance(v, (list, tuple, set)):
                v = list(v)
                t = ', '.join(_fmt_valor(x) for x in v[:8])
                return f"[{t}{', … (+%d)' % (len(v) - 8) if len(v) > 8 else ''}]"
            t = str(v)
            return t if len(t) <= 80 else t[:80] + '…'
        except Exception:
            return '?'

    def _firma_df(df):
        try:
            return (tuple(map(str, df.columns)), len(df),
                    int(pd.util.hash_pandas_object(df.astype(str), index=True).sum()))
        except Exception:
            return None

    def _anotar_cambio(ident, valor, texto_cambio, menu_forzado=None):
        """Compara con el valor de la corrida anterior y registra el cambio. Si el widget no se
        dibujo en la corrida previa (aparece por primera vez o reaparece) solo fija la base."""
        ss = st.session_state
        base = ss.setdefault('_log_valores', {})
        run_n = ss.get('_log_run_n', 0)
        previo = base.get(ident)
        base[ident] = (valor, run_n)
        if previo is None or previo[1] < run_n - 1:
            return
        try:
            cambio = bool(previo[0] != valor)
        except Exception:
            cambio = False
        if cambio:
            if menu_forzado:  # el cambio se atribuye a otro menu (p. ej. selector de resultados -> M17)
                menu_prev = _LOGS.menu
                _LOGS.menu = menu_forzado
                try:
                    _LOGS.log('INFO', texto_cambio(previo[0], valor))
                finally:
                    _LOGS.menu = menu_prev
            else:
                _LOGS.log('INFO', texto_cambio(previo[0], valor))

    def _limpio(label, key):
        t = re.sub(r'\*\*|`|:material/\w+:', '', str(label if label not in (None, '') else (key or '?'))).strip()
        return t or str(key or '?')

    def _envolver(owner, nombre, es_metodo, tipo, extra=None):
        orig = getattr(owner, nombre, None)
        while orig is not None and getattr(orig, '_sim_log_wrapper', False):  # evita envolver dos veces
            orig = getattr(orig, '__wrapped__', None)
        if orig is None:
            return

        def core(selfargs, a, k):
            if _LOGS.depth:  # llamada anidada (st.x -> DeltaGenerator.x): solo se registra una vez
                return orig(*selfargs, *a, **k)
            _LOGS.depth += 1
            try:
                r = orig(*selfargs, *a, **k)
            finally:
                _LOGS.depth -= 1
            try:
                primero = a[0] if a else k.get('label', k.get('body', k.get('data')))
                key = k.get('key')
                if str(key or '').startswith('_log_'):
                    return r
                if tipo == 'msg':
                    _LOGS.log(extra, (str(primero)[:160] + '…') if extra == 'INFO' and len(str(primero)) > 160 else primero,
                              dedupe=True)
                elif tipo == 'vista':
                    _registrar_vista(extra, a[0] if a else k.get('figure_or_data', k.get('data')))
                elif tipo == 'btn':
                    if r:
                        _LOGS.log('INFO', f"{extra}: {_limpio(primero, key)}")
                        _logs_refrescar()
                elif tipo == 'valor':
                    nom = _limpio(primero, key)
                    base_id = f"{_LOGS.menu}|{key if key is not None else nom}"
                    ident = base_id if key is not None else f"{base_id}|#{_ocurrencia(base_id)}"
                    _anotar_cambio(
                        ident, r,
                        lambda a_, n_: (f"Cambió «{nom}»: {_fmt_valor(a_)} → {_fmt_valor(n_)}"
                                        if not isinstance(n_, bool)
                                        else f"«{nom}» {'activado' if n_ else 'desactivado'}"),
                        _FLUJO_MENU_POR_KEY.get(str(key)) if key is not None else None)
                elif tipo == 'tabla':
                    firma = _firma_df(r)
                    if firma is not None:
                        nom = _limpio(None, key or 'tabla')
                        base_id = f"{_LOGS.menu}|tabla|{key if key is not None else nom}"
                        ident = base_id if key is not None else f"{base_id}|#{_ocurrencia(base_id)}"
                        _anotar_cambio(ident, firma, lambda a_, n_: f"Tabla «{nom}» editada ({n_[1]} filas)")
            except Exception:
                pass
            return r

        if es_metodo:
            @functools.wraps(orig)
            def w(self, *a, **k):
                return core((self,), a, k)
        else:
            @functools.wraps(orig)
            def w(*a, **k):
                return core((), a, k)
        w._sim_log_wrapper = True
        setattr(owner, nombre, w)

    _MENSAJES = (('success', 'SUCCESS'), ('info', 'INFO'), ('warning', 'WARNING'), ('error', 'ERROR'), ('toast', 'INFO'))
    _BOTONES = (('button', 'Botón pulsado'), ('download_button', 'Descarga solicitada'),
                ('form_submit_button', 'Formulario enviado'))
    _VALORES = ('number_input', 'text_input', 'text_area', 'selectbox', 'multiselect', 'radio', 'checkbox',
                'toggle', 'slider', 'select_slider', 'date_input', 'time_input', 'segmented_control', 'pills',
                'color_picker')
    # Se parchea tanto el modulo (st.xxx) como la clase (columna.xxx, contenedor.xxx, status_text.xxx).
    for owner, es_metodo in ((st, False), (_DG, True)):
        for nombre, nivel in _MENSAJES:
            _envolver(owner, nombre, es_metodo, 'msg', nivel)
        for nombre, verbo in _BOTONES:
            _envolver(owner, nombre, es_metodo, 'btn', verbo)
        for nombre in _VALORES:
            _envolver(owner, nombre, es_metodo, 'valor')
        _envolver(owner, 'data_editor', es_metodo, 'tabla')
        for nombre in ('plotly_chart', 'pyplot', 'altair_chart', 'line_chart', 'bar_chart', 'area_chart'):
            _envolver(owner, nombre, es_metodo, 'vista', 'grafico')
        for nombre in ('dataframe', 'table'):
            _envolver(owner, nombre, es_metodo, 'vista', 'tabla')

    orig_exc = getattr(st, 'exception', None)
    if orig_exc is not None:
        def w_exc(exception, *a, **k):
            _LOGS.log('ERROR', f"{type(exception).__name__}: {exception}", dedupe=True)
            return orig_exc(exception, *a, **k)
        st.exception = w_exc

    orig_up = getattr(st, 'file_uploader', None)
    if orig_up is not None:
        def w_up(label, *a, **k):
            r = orig_up(label, *a, **k)
            try:
                archivos = r if isinstance(r, list) else ([r] if r is not None else [])
                vistos = st.session_state.setdefault('_log_archivos_vistos', set())
                for f in archivos:
                    clave = (k.get('key'), f.name, f.size)
                    if clave not in vistos:
                        vistos.add(clave)
                        _LOGS.log('INFO', f"Archivo cargado: {f.name} ({f.size / 1024:.1f} KB)")
            except Exception:
                pass
            return r
        st.file_uploader = w_up

    _show_orig = _warnings.showwarning
    def _show(message, category, filename, lineno, file=None, line=None):
        _LOGS.log('WARNING', f"{category.__name__}: {message} ({os.path.basename(str(filename))}:{lineno})")
        return _show_orig(message, category, filename, lineno, file, line)
    _warnings.showwarning = _show


_LOGS_PLANTILLA_HTML = r'''<div id="sim-log-anchor" style="display:none"></div>
<script>
(function () {
  var W, D;
  try { W = (window.parent && window.parent.document) ? window.parent : window; D = W.document; }
  catch (e) { W = window; D = document; }
  var VER = '__VER__';
  var DATA = __DATA__;

  // Oculta el contenedor vacio que Streamlit reserva para este bloque (evita un hueco en la pagina).
  function ocultar() {
    try {
      var a = document.getElementById('sim-log-anchor');
      var c = a && a.closest && a.closest('[data-testid="stElementContainer"]');
      if (c) c.style.display = 'none';
    } catch (e) {}
  }
  ocultar(); setTimeout(ocultar, 60);

    function addResize(win, P, save, minW, minH) {
    ['n', 's', 'e', 'w', 'ne', 'nw', 'se', 'sw'].forEach(function (dir) {
      var hdl = D.createElement('div');
      hdl.className = 'sim-rz sim-rz-' + dir;
      hdl.addEventListener('pointerdown', function (e) {
        var r = win.getBoundingClientRect();
        var sx = e.clientX, sy = e.clientY, sl = r.left, stp = r.top, sw = r.width, sh = r.height;
        var vw = W.innerWidth, vh = W.innerHeight;
        try { hdl.setPointerCapture(e.pointerId); } catch (er) {}
        function mv(ev) {
          var dx = ev.clientX - sx, dy = ev.clientY - sy;
          var l = sl, t = stp, w = sw, h = sh;
          if (dir.indexOf('e') >= 0) { w = Math.max(minW, Math.min(sw + dx, vw - sl)); }
          if (dir.indexOf('s') >= 0) { h = Math.max(minH, Math.min(sh + dy, vh - stp)); }
          if (dir.indexOf('w') >= 0) {
            w = Math.max(minW, sw - dx); l = sl + sw - w;
            if (0 > l) { w = w + l; l = 0; }
          }
          if (dir.indexOf('n') >= 0) {
            h = Math.max(minH, sh - dy); t = stp + sh - h;
            if (0 > t) { h = h + t; t = 0; }
          }
          win.style.left = l + 'px'; win.style.top = t + 'px';
          win.style.width = w + 'px'; win.style.height = h + 'px';
        }
        function up() {
          hdl.removeEventListener('pointermove', mv);
          hdl.removeEventListener('pointerup', up);
          hdl.removeEventListener('pointercancel', up);
          P.x = parseInt(win.style.left, 10); P.y = parseInt(win.style.top, 10);
          P.w = parseInt(win.style.width, 10); P.h = parseInt(win.style.height, 10);
          save();
        }
        hdl.addEventListener('pointermove', mv);
        hdl.addEventListener('pointerup', up);
        hdl.addEventListener('pointercancel', up);
        e.preventDefault(); e.stopPropagation();
      });
      win.appendChild(hdl);
    });
  }

  function initUI(S) {
    var KEY = 'simLogPrefs_v1';
    var P = { open: true, x: null, y: null, w: 620, h: 300, f: { INFO: 1, SUCCESS: 1, WARNING: 1, ERROR: 1 }, cleared: 0 };
    try {
      var raw = W.localStorage.getItem(KEY);
      if (raw) { var o = JSON.parse(raw); for (var k in o) { P[k] = o[k]; } }
    } catch (e) {}
    function save() { try { W.localStorage.setItem(KEY, JSON.stringify(P)); } catch (e) {} }
    function grp(l) { return l === 'CRITICAL' ? 'ERROR' : (l === 'DEBUG' ? 'INFO' : l); }
    var COL = { INFO: '#7aa2f7', SUCCESS: '#7ee787', WARNING: '#e3b341', ERROR: '#ff7b72' };

    var css = D.createElement('style');
    css.id = 'sim-log-style';
    css.textContent =
      '#sim-log-root{font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;}' +
      '#sim-log-btn{position:fixed;right:18px;bottom:18px;z-index:1000001;background:#1e1e1e;color:#d4d4d4;' +
      'border:1px solid #3c3c3c;border-radius:20px;padding:7px 14px;cursor:pointer;display:flex;gap:8px;align-items:center;' +
      'font:600 12px ui-monospace,Menlo,Consolas,monospace;box-shadow:0 2px 10px rgba(0,0,0,.35);}' +
      '#sim-log-btn:hover{background:#2a2a2a;}' +
      '#sim-log-btn .bdg{border-radius:9px;padding:0 7px;font-size:11px;color:#111;font-weight:700;}' +
      '#sim-log-win{position:fixed;z-index:1000000;background:#1e1e1e;color:#d4d4d4;border:1px solid #454545;' +
      'border-radius:8px;box-shadow:0 8px 30px rgba(0,0,0,.45);display:flex;flex-direction:column;overflow:hidden;' +
      'min-width:340px;min-height:150px;box-sizing:border-box;}' +
      '.sim-rz{position:absolute;z-index:10;touch-action:none;}' +
      '.sim-rz-n{top:0;left:10px;right:10px;height:6px;cursor:ns-resize;}' +
      '.sim-rz-s{bottom:0;left:10px;right:10px;height:6px;cursor:ns-resize;}' +
      '.sim-rz-e{right:0;top:10px;bottom:10px;width:6px;cursor:ew-resize;}' +
      '.sim-rz-w{left:0;top:10px;bottom:10px;width:6px;cursor:ew-resize;}' +
      '.sim-rz-ne{top:0;right:0;width:12px;height:12px;cursor:nesw-resize;}' +
      '.sim-rz-nw{top:0;left:0;width:12px;height:12px;cursor:nwse-resize;}' +
      '.sim-rz-sw{bottom:0;left:0;width:12px;height:12px;cursor:nesw-resize;}' +
      '.sim-rz-se{bottom:0;right:0;width:16px;height:16px;cursor:nwse-resize;background:linear-gradient(135deg,rgba(0,0,0,0) 50%,#8b949e 50%,#8b949e 60%,rgba(0,0,0,0) 60%,rgba(0,0,0,0) 72%,#8b949e 72%,#8b949e 82%,rgba(0,0,0,0) 82%);}' +
      '#sim-log-hd{display:flex;align-items:center;gap:6px;padding:6px 10px;background:#2d2d30;cursor:move;' +
      'user-select:none;-webkit-user-select:none;border-bottom:1px solid #3c3c3c;flex:0 0 auto;touch-action:none;flex-wrap:wrap;}' +
      '#sim-log-hd .ttl{font-weight:700;color:#fff;margin-right:auto;white-space:nowrap;}' +
      '#sim-log-hd button{background:#3a3d41;color:#d4d4d4;border:1px solid #4a4d52;border-radius:4px;padding:1px 8px;' +
      'cursor:pointer;font:inherit;line-height:1.5;}' +
      '#sim-log-hd button:hover{background:#4a4d52;}' +
      '#sim-log-hd button.chip{font-weight:700;opacity:.4;}' +
      '#sim-log-hd button.chip.on{opacity:1;}' +
      '#sim-log-bd{flex:1 1 auto;min-height:0;overflow:auto;background:#1e1e1e;}' +
      '#sim-log-bd table{width:100%;border-collapse:collapse;}' +
      '#sim-log-bd th{position:sticky;top:0;background:#4a5561;color:#fff;text-align:left;padding:5px 10px;font-weight:600;}' +
      '#sim-log-bd td{padding:3px 10px;border-bottom:1px solid #2a2a2a;vertical-align:top;}' +
      '#sim-log-bd td.t{white-space:nowrap;color:#8b949e;}' +
      '#sim-log-bd td.l{white-space:nowrap;font-weight:700;}' +
      '#sim-log-bd td.m{white-space:pre-wrap;word-break:break-word;}' +
      '#sim-log-ft{flex:0 0 auto;padding:3px 24px 3px 10px;background:#252526;color:#8b949e;border-top:1px solid #3c3c3c;font-size:11px;}';
    D.head.appendChild(css);

    var root = D.createElement('div'); root.id = 'sim-log-root';
    var win = D.createElement('div'); win.id = 'sim-log-win';
    var hd = D.createElement('div'); hd.id = 'sim-log-hd';
    var ttl = D.createElement('span'); ttl.className = 'ttl'; ttl.textContent = '\u25A0 Registro de actividad';
    hd.appendChild(ttl);

    var q = D.createElement('input'); q.type = 'text'; q.placeholder = 'Buscar… (ej. M5, Redis)'; q.value = P.q || '';
    q.style.cssText = 'background:#3a3d41;color:#d4d4d4;border:1px solid #4a4d52;border-radius:4px;padding:1px 8px;font:inherit;width:150px;';
    ['keydown', 'keyup', 'keypress'].forEach(function (ev) { q.addEventListener(ev, function (e) { e.stopPropagation(); }); });
    q.addEventListener('input', function () { P.q = q.value.toLowerCase(); save(); S.render(); });
    hd.appendChild(q);

    var chips = {};
    ['INFO', 'SUCCESS', 'WARNING', 'ERROR'].forEach(function (lv) {
      var b = D.createElement('button'); b.className = 'chip'; b.textContent = lv; b.style.color = COL[lv];
      b.title = 'Mostrar/ocultar ' + lv;
      b.addEventListener('click', function () { P.f[lv] = P.f[lv] ? 0 : 1; save(); S.render(); });
      chips[lv] = b; hd.appendChild(b);
    });
    function mkBtn(txt, title, fn) {
      var b = D.createElement('button'); b.textContent = txt; b.title = title;
      b.addEventListener('click', fn); hd.appendChild(b); return b;
    }
    mkBtn('Limpiar', 'Oculta lo registrado hasta ahora', function () {
      var d = S.data; P.cleared = d.length ? d[d.length - 1][0] : 0; save(); S.render();
    });
    mkBtn('Descargar', 'Descargar los logs visibles (.txt)', function () { S.download(); });
    mkBtn('\u2715', 'Ocultar ventana (Ctrl + `)', function () { S.setOpen(false); });

    var bd = D.createElement('div'); bd.id = 'sim-log-bd';
    var table = D.createElement('table');
    var thead = D.createElement('thead'); var trh = D.createElement('tr');
    ['Time', 'Level', 'Message'].forEach(function (t) { var th = D.createElement('th'); th.textContent = t; trh.appendChild(th); });
    thead.appendChild(trh); table.appendChild(thead);
    var tbody = D.createElement('tbody'); table.appendChild(tbody); bd.appendChild(table);
    var ft = D.createElement('div'); ft.id = 'sim-log-ft';
    win.appendChild(hd); win.appendChild(bd); win.appendChild(ft);

    var btn = D.createElement('div'); btn.id = 'sim-log-btn'; btn.title = 'Mostrar/ocultar logs (Ctrl + `)';
    var btnTxt = D.createElement('span'); btnTxt.textContent = '\u25A3 Logs';
    var badge = D.createElement('span'); badge.className = 'bdg';
    btn.appendChild(btnTxt); btn.appendChild(badge);
    btn.addEventListener('click', function () { S.setOpen(!P.open); });

    root.appendChild(win); root.appendChild(btn); D.body.appendChild(root);
    S.root = root;
    addResize(win, P, save, 340, 150);

    function place() {
      var vw = W.innerWidth, vh = W.innerHeight;
      var w = Math.min(P.w, vw - 20), h = Math.min(P.h, vh - 80);
      win.style.width = w + 'px'; win.style.height = h + 'px';
      var x = (P.x === null || P.x === undefined) ? vw - w - 18 : P.x;
      var y = (P.y === null || P.y === undefined) ? vh - h - 64 : P.y;
      x = Math.max(0, Math.min(x, vw - 80)); y = Math.max(0, Math.min(y, vh - 40));
      win.style.left = x + 'px'; win.style.top = y + 'px';
    }

    // ---- arrastrar la ventana desde la barra de titulo ----
    hd.addEventListener('pointerdown', function (e) {
      if (e.target && e.target.closest && e.target.closest('button')) return;
      var r = win.getBoundingClientRect(); var dx = e.clientX - r.left, dy = e.clientY - r.top;
      try { hd.setPointerCapture(e.pointerId); } catch (er) {}
      function mv(ev) {
        var x = Math.max(0, Math.min(ev.clientX - dx, W.innerWidth - 80));
        var y = Math.max(0, Math.min(ev.clientY - dy, W.innerHeight - 40));
        win.style.left = x + 'px'; win.style.top = y + 'px';
      }
      function up() {
        hd.removeEventListener('pointermove', mv); hd.removeEventListener('pointerup', up);
        P.x = parseInt(win.style.left, 10); P.y = parseInt(win.style.top, 10); save();
      }
      hd.addEventListener('pointermove', mv); hd.addEventListener('pointerup', up);
      e.preventDefault();
    });

    // ---- recordar el tamano cuando el usuario lo cambia ----
    if (W.ResizeObserver) {
      new W.ResizeObserver(function () {
        if (win.offsetWidth > 0 && win.offsetHeight > 0 && P.open) { P.w = win.offsetWidth; P.h = win.offsetHeight; save(); }
      }).observe(win);
    }
    W.addEventListener('resize', function () { if (P.open) place(); });

    S.kd = function (e) {
      if (e.ctrlKey && (e.key === '`' || e.code === 'Backquote')) { e.preventDefault(); S.setOpen(!P.open); }
    };
    D.addEventListener('keydown', S.kd);

    S.setOpen = function (v) {
      P.open = !!v; save();
      win.style.display = P.open ? 'flex' : 'none';
      if (P.open) { place(); S.render(true); }
    };

    function visibles() {
      return S.data.filter(function (r) {
        return r[0] > P.cleared && P.f[grp(r[2])] && (!P.q || (r[3] + ' ' + r[2]).toLowerCase().indexOf(P.q) >= 0);
      });
    }
    S.download = function () {
      var txt = visibles().map(function (r) { return r[1] + ' | ' + (r[2] + '        ').slice(0, 8) + ' | ' + r[3]; }).join('\n');
      var blob = new Blob([txt || '(sin registros)'], { type: 'text/plain;charset=utf-8' });
      var a = D.createElement('a'); var url = W.URL.createObjectURL(blob);
      a.href = url; a.download = 'logs_simulador.txt'; D.body.appendChild(a); a.click();
      setTimeout(function () { D.body.removeChild(a); W.URL.revokeObjectURL(url); }, 500);
    };

    S.render = function (forzarAbajo) {
      var d = S.data, mx = d.length ? d[d.length - 1][0] : 0;
      if (P.cleared > mx) { P.cleared = 0; save(); }  // el servidor se reinicio: la cuenta empieza de nuevo
      var vis = visibles();
      var cerca = (bd.scrollHeight - bd.scrollTop - bd.clientHeight) < 50;
      var primera = !S.rendered;
      var frag = D.createDocumentFragment();
      if (!vis.length) {
        var tr0 = D.createElement('tr'); var td0 = D.createElement('td');
        td0.colSpan = 3; td0.style.textAlign = 'center'; td0.style.opacity = '.6';
        td0.textContent = d.length ? 'Sin registros para los filtros actuales.' : 'Sin registros todavia.';
        tr0.appendChild(td0); frag.appendChild(tr0);
      }
      vis.forEach(function (r) {
        var tr = D.createElement('tr');
        var t = D.createElement('td'); t.className = 't'; t.textContent = r[1];
        var l = D.createElement('td'); l.className = 'l'; l.textContent = r[2]; l.style.color = COL[grp(r[2])] || '#d4d4d4';
        var m = D.createElement('td'); m.className = 'm'; m.textContent = r[3];
        tr.appendChild(t); tr.appendChild(l); tr.appendChild(m); frag.appendChild(tr);
      });
      tbody.textContent = ''; tbody.appendChild(frag);
      if (cerca || primera || forzarAbajo === true) { bd.scrollTop = bd.scrollHeight; }
      S.rendered = true;

      for (var lv in chips) { chips[lv].className = 'chip' + (P.f[lv] ? ' on' : ''); }
      var nErr = 0, nWar = 0, nTot = 0;
      d.forEach(function (r) {
        if (r[0] > P.cleared) { nTot++; var g = grp(r[2]); if (g === 'ERROR') nErr++; else if (g === 'WARNING') nWar++; }
      });
      badge.textContent = nErr ? (nErr + ' err') : (nWar ? (nWar + ' warn') : String(nTot));
      badge.style.background = nErr ? '#ff7b72' : (nWar ? '#e3b341' : '#7aa2f7');
      ft.textContent = vis.length + ' de ' + nTot + ' registros \u00B7 arrastra el titulo para mover, esquina inferior derecha para cambiar tamano \u00B7 Ctrl + ` muestra/oculta';
    };

    place();
    win.style.display = P.open ? 'flex' : 'none';
  }

  var S = W.__simLog;
  if (!S || S.ver !== VER) {
    if (S) {
      try { if (S.root && S.root.parentNode) S.root.parentNode.removeChild(S.root); } catch (e) {}
      try { if (S.kd) D.removeEventListener('keydown', S.kd); } catch (e) {}
      try { var oldCss = D.getElementById('sim-log-style'); if (oldCss) oldCss.parentNode.removeChild(oldCss); } catch (e) {}
    }
    S = W.__simLog = { ver: VER, data: [] };
    initUI(S);
  }
  S.data = DATA;
  S.render();
})();
</script>
'''


def _logs_refrescar(force=False, gap=0.5):
    """Redibuja la ventana flotante de logs. Se puede llamar en cualquier punto del script (por ej.
    durante la simulacion) para que el usuario vea el avance en vivo. Limitada a ~2 veces por segundo."""
    try:
        slot = getattr(_LOGS, 'slot', None)
        if slot is None:
            return
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is None:  # hilo de trabajo: Streamlit no permite dibujar desde ahi
            return
        ahora = time.time()
        if not force and ahora - getattr(_LOGS, 'ultimo_refresco', 0.0) < gap:
            return
        _LOGS.ultimo_refresco = ahora
        with _LOGS.lock:
            filas = [list(f) if len(f) == 4 else [0] + list(f) for f in list(_LOGS.items)[-500:]]
        datos = (json.dumps(filas, ensure_ascii=False).replace('<', '\\u003c')
                 .replace('\u2028', '\\u2028').replace('\u2029', '\\u2029'))
        html = _LOGS_PLANTILLA_HTML.replace('__VER__', 'v3').replace('__DATA__', datos)
        try:
            slot.html(html, unsafe_allow_javascript=True)
        except TypeError:  # Streamlit antiguo sin unsafe_allow_javascript: se usa un iframe sin altura
            import streamlit.components.v1 as components
            with slot:
                components.html(html, height=0)
        _flujo_ref = globals().get('_flujo_refrescar')
        if _flujo_ref is not None:
            _flujo_ref(force=True)
    except Exception:
        pass


# ======================================================================
# #####VENTANA FLOTANTE DE FLUJO DE DATOS (diagrama de menus)#####
# ======================================================================
# Segunda ventana flotante (fondo oscuro, arrastrable, redimensionable, con scroll y zoom), igual en
# espiritu a la de logs. Dibuja los menus como nodos agrupados por etapa (Datos, Configuracion,
# Restricciones y eventos, Ejecucion y resultados) unidos por conectores, y colorea cada nodo segun lo
# que fue ocurriendo: exito, advertencia, error, cambios del usuario u OMITIDO (menu que el usuario no
# toco antes de simular). Ademas dibuja la "ruta del usuario" (por que menus fue pasando, en orden).
# Se alimenta SOLO desde log_evento (no toca la logica de la app): cada mensaje ya viene etiquetado con
# el menu en el que ocurrio. El estado vive en _LOGS (sobrevive a los reruns de Streamlit).
_FLUJO_MENUS_OPCIONALES = ('5', '6', '7', '8', '9', '10', '11', '12', '13', '14')
_FLUJO_PREFIJOS_ACCION = ('Cambió', 'Botón pulsado', 'Descarga solicitada', 'Formulario enviado',
                          'Archivo cargado', 'Tabla «')


def _flujo_estado():
    """Estado del diagrama de flujo (se crea la primera vez; vive en el modulo falso _LOGS)."""
    f = getattr(_LOGS, 'flujo', None)
    if f is None:
        f = {'nodos': {}, 'ruta': [], 'ultimo': None, 'activo': None, 'seq': 0}
        _LOGS.flujo = f
    return f


_FLUJO_PREFIJOS_VISTA = ('Visualizó',)
_FLUJO_MENUS_RESULTADO = ('17', '17b', '18', '19', '20', '20b', '21', '22')
_FLUJO_MENUS_ENTRADA_SIM = ('1', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15')


def _flujo_nodo(f, mid, titulo):
    n = f['nodos'].get(mid)
    if n is None:
        n = {'titulo': titulo, 'estado': 'pendiente', 'visto': False, 'tocado': False,
             'ok': 0, 'warn': 0, 'err': 0, 'info': 0, 'ultimo': '', 'ts': '',
             'eventos': [], 'progreso': None, 'seq': 0}
        f['nodos'][mid] = n
    elif titulo:
        n['titulo'] = titulo
    for k_, v_ in (('cambios', 0), ('vistas', 0), ('obsoleto', False), ('run', -1), ('vig', []), ('omitido', False),
                 ('dur', None), ('dur_accion', None), ('dur_proc', None), ('accion_run', -1)):
        n.setdefault(k_, v_)
    return n


def _flujo_es_accion(texto):
    """True si el mensaje corresponde a algo que HIZO el usuario (boton, cambio de valor, archivo...),
    y no a un aviso automatico que la app vuelve a dibujar."""
    return texto.startswith(_FLUJO_PREFIJOS_ACCION) or bool(re.match(r'^«.*» (activado|desactivado)$', texto))


def _flujo_mid_titulo(menu):
    m = re.match(r'^M(\S+) · (.*)$', menu or '')
    return (m.group(1), m.group(2)) if m else ('SYS', 'Sistema')


def _flujo_menu_render(numero, titulo):
    """Se llama cada vez que un menu se DIBUJA en esta corrida. Devuelve True si el menu aparece por primera
    vez o reaparece despues de haber desaparecido (p. ej. al quitar un archivo y volver a cargarlo)."""
    try:
        c = getattr(_LOGS, 'run_id', 0)
        with _LOGS.lock:
            f = _flujo_estado()
            n = _flujo_nodo(f, str(numero), titulo)
            nuevo = (not n['visto']) or n['run'] < c - 1
            if n['run'] != c:
                n['run'] = c
                n['vig'] = []  # los mensajes vigentes se reconstruyen en cada corrida
        return nuevo
    except Exception:
        return False


def _flujo_menu_duracion(numero, titulo, segundos):
    """Guarda cuanto tardo en ejecutarse la seccion de un menu en esta corrida ('dur'). Si en esa misma corrida
    el usuario hizo una accion en el menu (cambio, boton, archivo), tambien se guarda como 'dur_accion': el
    tiempo que tomo procesar ese ultimo cambio (se conserva aunque despues otras corridas actualicen 'dur')."""
    try:
        c = getattr(_LOGS, 'run_id', 0)
        with _LOGS.lock:
            f = _flujo_estado()
            n = _flujo_nodo(f, str(numero), titulo)
            n['dur'] = float(segundos)
            if n.get('accion_run') == c:
                n['dur_accion'] = float(segundos)
    except Exception:
        pass


def _flujo_msg_vigente(nivel, texto, menu):
    """Marca un mensaje persistente (st.success/info/warning/error) como vigente en ESTA corrida. Si el mensaje
    deja de dibujarse (el usuario corrigio algo), el menu deja de arrastrar ese estado."""
    try:
        if texto.startswith('Simulación reiniciada'):
            nivel = 'INFO'
        mid, titulo = _flujo_mid_titulo(menu)
        c = getattr(_LOGS, 'run_id', 0)
        with _LOGS.lock:
            f = _flujo_estado()
            n = _flujo_nodo(f, mid, titulo)
            if n['run'] != c:
                n['run'] = c
                n['vig'] = []
            par = [nivel, texto[:160]]
            if par not in n['vig']:
                n['vig'].append(par)
                del n['vig'][:-8]
    except Exception:
        pass


def _flujo_registrar(nivel, texto, menu, persistente=False):
    """Anota un evento en el nodo del menu donde ocurrio (o en 'SYS' si fue fuera de todo menu).
    Aqui solo se acumula la HISTORIA (contadores, eventos, cambios del usuario, ruta, omitidos, estado de la
    simulacion). El color/estado que se muestra se calcula en _flujo_vista_nodos a partir de lo que esta
    vigente en la corrida actual, para que nunca quede un estado viejo."""
    try:
        if texto.startswith('Simulación reiniciada'):
            nivel = 'INFO'
        mid, titulo = _flujo_mid_titulo(menu)
        ts = _datetime.now().strftime('%H:%M:%S')
        c = getattr(_LOGS, 'run_id', 0)
        with _LOGS.lock:
            f = _flujo_estado()
            n = _flujo_nodo(f, mid, titulo)
            f['seq'] += 1
            n['seq'] = f['seq']
            n['visto'] = True
            n['ts'] = ts
            f['activo'] = mid
            if texto == 'Sección mostrada':
                n['eventos'].append([ts, nivel, texto])
                del n['eventos'][:-12]
                return
            if not persistente:  # los mensajes persistentes viven en 'vig' (vigentes), no como "ultimo evento"
                n['ultimo'] = texto[:160]
            accion = _flujo_es_accion(texto)
            vista = texto.startswith(_FLUJO_PREFIJOS_VISTA)
            if nivel == 'SUCCESS':
                n['ok'] += 1
            elif nivel == 'WARNING':
                n['warn'] += 1
            elif nivel in ('ERROR', 'CRITICAL'):
                n['err'] += 1
            else:
                n['info'] += 1
            if accion:
                n['tocado'] = True
                n['omitido'] = False
                n['cambios'] += 1
                n['accion_run'] = c
            elif vista:
                n['vistas'] += 1
            n['eventos'].append([ts, nivel, texto[:220]])
            del n['eventos'][:-12]

            pm = re.match(r'^Simulación \[(.*?)\] (\d+)%', texto)
            if pm and mid == '16':
                n['progreso'] = int(pm.group(2))

            # Ruta del usuario: acciones propias y, en los menus de resultados, lo que el usuario visualiza.
            if (accion or (vista and mid in _FLUJO_MENUS_RESULTADO)) and mid != f['ultimo']:
                if f['ultimo'] is not None:
                    f['ruta'].append([f['ultimo'], mid, f['seq']])
                    del f['ruta'][:-40]
                f['ultimo'] = mid

            # Estado de la simulacion (M16): en curso / completada / fallida.
            if mid == '16':
                if texto.startswith('Simulación iniciada'):
                    f['sim_run'] = True
                    f['sim_t0'] = time.perf_counter()
                    n['dur_proc'] = None
                    f['sim_error'] = False
                elif nivel == 'SUCCESS' and texto.startswith('Simulación completada'):
                    f['sim_run'] = False
                    f['sim_error'] = False
                    f['sim_ok'] = True
                    if f.get('sim_t0') is not None:
                        n['dur_proc'] = time.perf_counter() - f['sim_t0']
                    n['obsoleto'] = False
                    for k in _FLUJO_MENUS_RESULTADO:
                        otro = f['nodos'].get(k)
                        if otro is not None:
                            otro['obsoleto'] = False
                elif nivel in ('ERROR', 'CRITICAL') and texto.startswith('La simulación falló'):
                    f['sim_run'] = False
                    f['sim_error'] = True
                    if f.get('sim_t0') is not None:
                        n['dur_proc'] = time.perf_counter() - f['sim_t0']
                    f['sim_ok'] = False

            # Si el usuario cambia datos/configuracion DESPUES de simular, los resultados quedan desactualizados.
            if f.get('sim_ok') and ((accion and (mid in _FLUJO_MENUS_ENTRADA_SIM or (mid == '16' and texto.startswith('Cambió'))))
                                    or texto.startswith('Simulación reiniciada')):
                f['sim_ok'] = False
                for k in _FLUJO_MENUS_RESULTADO:
                    otro = f['nodos'].get(k)
                    if otro is not None and otro['visto']:
                        otro['obsoleto'] = True
                        otro['eventos'].append([ts, 'WARNING', f'Resultados desactualizados: cambio en M{mid}; hay que volver a simular.'])
                        del otro['eventos'][:-12]

            # Al pulsar SIMULAR, los menus de configuracion visibles que el usuario nunca toco quedan como OMITIDOS
            # (la simulacion usa sus valores por defecto).
            if mid == '16' and texto.startswith('Botón pulsado') and 'SIMULAR' in texto.upper():
                for k in _FLUJO_MENUS_OPCIONALES:
                    otro = f['nodos'].get(k)
                    if otro is not None and otro['run'] == c and not otro['tocado'] and not otro['omitido']:
                        otro['omitido'] = True
                        otro['ts'] = ts
                        otro['eventos'].append([ts, 'OMITIDO', 'El usuario no modifico este menu antes de simular: se usaron los valores por defecto.'])
                        del otro['eventos'][:-12]
    except Exception:
        pass


def _flujo_vista_nodos(f):
    """Estado que se DIBUJA, calculado con lo vigente ahora:
    - menu no dibujado en la corrida actual -> pendiente (p. ej. se quito un archivo: todo lo que dependia vuelve a pendiente)
    - mensajes vigentes: error > advertencia
    - M1 es OK solo si los datos estan cargados y validados; M16 refleja la simulacion (en curso / completada / fallida)
    - menu editado por el usuario -> EDIT; omitido al simular -> OMIT; si no, OK (se muestra bien y sin problemas).
    Mientras la corrida esta en curso, los menus que aun no se dibujan conservan lo de la corrida anterior."""
    c = getattr(_LOGS, 'run_id', 0)
    fin = bool(f.get('fin'))
    try:
        listo1 = bool(st.session_state.get('datos_cargados'))
    except Exception:
        listo1 = False
    out = {}
    for mid, n in f['nodos'].items():
        d = dict(n, eventos=[list(e) for e in n['eventos']], vig=[list(v) for v in n.get('vig', [])])
        presente = n.get('run') == c or (not fin and n.get('run') == c - 1)
        niveles = [v[0] for v in d['vig']] if presente else []
        d['warn'] = niveles.count('WARNING')
        d['err'] = niveles.count('ERROR') + niveles.count('CRITICAL')
        d['presente'] = presente
        d['obsoleto'] = bool(n.get('obsoleto')) and presente
        if not presente:
            est = 'pendiente'
            d['vig'] = []
        elif d['err']:
            est = 'error'
        elif d['warn']:
            est = 'advertencia'
        elif mid == '16':
            est = ('error' if f.get('sim_error') else 'activo' if f.get('sim_run')
                   else 'exito' if f.get('sim_ok') else 'pendiente')
        elif mid == '1':
            est = 'exito' if listo1 else 'pendiente'
        elif mid in ('2', '3'):
            est = 'exito'
        elif n.get('omitido') and not n.get('tocado'):
            est = 'omitido'
        else:
            est = 'exito'
        d['estado'] = est
        d['editado'] = bool(n.get('tocado')) and presente
        _usa_sim = (mid == '16' and n.get('dur_proc') is not None)
        _t = n.get('dur_proc') if _usa_sim else n.get('dur_accion')
        if _t is None:
            _t = n.get('dur')
        d['tiempo'] = _t if presente else None
        d['tiempo_tipo'] = 'sim' if _usa_sim else ''
        out[mid] = d
    return out


_FLUJO_PLANTILLA_HTML = r'''<div id="sim-flow-anchor" style="display:none"></div>
<script>
(function () {
  var W, D;
  try { W = (window.parent && window.parent.document) ? window.parent : window; D = W.document; }
  catch (e) { W = window; D = document; }
  var VER = '__VER__';
  var DATA = __DATA__;

  function ocultar() {
    try {
      var a = document.getElementById('sim-flow-anchor');
      var c = a && a.closest && a.closest('[data-testid="stElementContainer"]');
      if (c) c.style.display = 'none';
    } catch (e) {}
  }
  ocultar(); setTimeout(ocultar, 60);

  // ---------- geometria del diagrama ----------
  var NW = 200, NH = 68, STEP = 82, GW = 236, PX = 18;
  var GRUPOS = [
    { id: 'datos', t: 'Datos de entrada', c: '#35c9c3', x: 14, y: 30, n: ['1', '2', '3', '4'] },
    { id: 'config', t: 'Configuracion', c: '#a78bfa', x: 344, y: 30, n: ['5', '6', '7', '8', '9'] },
    { id: 'restr', t: 'Restricciones y eventos', c: '#5aa9ff', x: 674, y: 30, n: ['10', '11', '12', '13', '14', '15'] },
    { id: 'ejec', t: 'Ejecucion y resultados', c: '#ffb454', x: 1004, y: 30, n: ['16', '17', '17b', '18', '19', '20', '20b', '21', '22'] },
    { id: 'sys', t: 'Sistema', c: '#8b93a7', x: 14, y: 440, n: ['SYS'] }
  ];
  var CORTO = {
    '1': 'Cargar Datos', '2': 'Explorador', '3': 'Vaciaderos', '4': 'Nivel',
    '5': 'Parametros', '6': 'Recursos', '7': 'Flota por Turno', '8': 'Distribuciones', '9': 'Act. Personalizadas',
    '10': 'Rest. Geologicas', '11': 'Fallas de Equipos', '12': 'Mantenciones', '13': 'Frentes',
    '14': 'Rest. Fisicas', '15': 'Geometria',
    '16': 'Ejecutar Simulacion', '17': 'Avance Tuneles', '17b': 'Recursos y Flota', '18': 'Carta Gantt',
    '19': 'Res. Mantenciones', '20': 'Res. Fallas', '20b': 'Res. Geologia MWD', '21': 'Traza de Eventos', '22': 'Exportar Excel', 'SYS': 'Sistema'
  };
  // Conectores entre etapas (el flujo dentro de una etapa va encadenado, de arriba hacia abajo).
  var CRUCES = [
    ['4', '5', 'Nivel', 'Nivel seleccionado y tablas filtradas (frentes, actividades y recursos del nivel).'],
    ['9', '10', 'Config.', 'Parametros, recursos, distribuciones y actividades definidos pasan a las restricciones y eventos.'],
    ['15', '16', 'Geometria', 'Frentes seleccionados, restricciones fisicas y geometria llegan al motor de simulacion.']
  ];
  var COL = { pendiente: '#5b6178', visualizado: '#f0abfc', activo: '#38bdf8', exito: '#4ade80', advertencia: '#fbbf24', error: '#f87171', omitido: '#9aa3b8' };
  var COL_EDIT = '#5aa9ff';
  var ETQ = { pendiente: 'PENDIENTE', visualizado: 'VISTO', activo: 'EN CURSO', exito: 'EXITOSO', advertencia: 'ADVERTENCIA', error: 'ERROR', omitido: 'OMITIDO' };
  var NOMBRE_ESTADO = { pendiente: 'Pendiente', visualizado: 'Visualizado (sin cambios)', activo: 'En curso (simulando)', exito: 'Exitoso',
    advertencia: 'Con advertencia', error: 'Con error', omitido: 'Omitido (valores por defecto)' };
  var COL_NIVEL = { INFO: '#7aa2f7', SUCCESS: '#7ee787', WARNING: '#e3b341', ERROR: '#ff7b72', CRITICAL: '#ff7b72', OMITIDO: '#9aa3b8' };

  var POS = {}, GEO = [], VW = 0, VH = 0;
  GRUPOS.forEach(function (g) {
    var h = 38 + (g.n.length - 1) * STEP + NH + 14;
    GEO.push({ g: g, h: h });
    g.n.forEach(function (id, i) { POS[id] = { x: g.x + PX, y: g.y + 38 + i * STEP, c: g.c }; });
    VW = Math.max(VW, g.x + GW + 14);
    VH = Math.max(VH, g.y + h + 30);
  });

  function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
  function cortar(s, n) { s = String(s || ''); return s.length > n ? s.slice(0, n - 3) + '...' : s; }
  function fmtDur(s) {
    if (s === null || s === undefined || isNaN(s)) return '';
    if (s >= 60) { var m = Math.floor(s / 60); return m + ' min ' + Math.round(s - m * 60) + ' s'; }
    if (s >= 1) return s.toFixed(2) + ' s';
    if (s >= 0.001) return (s * 1000).toFixed(s >= 0.01 ? 0 : 1) + ' ms';
    return Math.max(1, Math.round(s * 1000000)) + ' µs';
  }

  function borde(c, t) {
    var dx = t.x - c.x, dy = t.y - c.y;
    var sx = dx ? (NW / 2 + 3) / Math.abs(dx) : Infinity;
    var sy = dy ? (NH / 2 + 3) / Math.abs(dy) : Infinity;
    var k = Math.min(sx, sy);
    return { x: c.x + dx * k, y: c.y + dy * k };
  }
  function dot(x, y, col) {
    return '<circle cx="' + x + '" cy="' + y + '" r="5" fill="#0b1220" stroke="' + col + '" stroke-width="2"/>' +
           '<circle cx="' + x + '" cy="' + y + '" r="1.8" fill="' + col + '"/>';
  }

    function addResize(win, P, save, minW, minH) {
    ['n', 's', 'e', 'w', 'ne', 'nw', 'se', 'sw'].forEach(function (dir) {
      var hdl = D.createElement('div');
      hdl.className = 'sim-rz sim-rz-' + dir;
      hdl.addEventListener('pointerdown', function (e) {
        var r = win.getBoundingClientRect();
        var sx = e.clientX, sy = e.clientY, sl = r.left, stp = r.top, sw = r.width, sh = r.height;
        var vw = W.innerWidth, vh = W.innerHeight;
        try { hdl.setPointerCapture(e.pointerId); } catch (er) {}
        function mv(ev) {
          var dx = ev.clientX - sx, dy = ev.clientY - sy;
          var l = sl, t = stp, w = sw, h = sh;
          if (dir.indexOf('e') >= 0) { w = Math.max(minW, Math.min(sw + dx, vw - sl)); }
          if (dir.indexOf('s') >= 0) { h = Math.max(minH, Math.min(sh + dy, vh - stp)); }
          if (dir.indexOf('w') >= 0) {
            w = Math.max(minW, sw - dx); l = sl + sw - w;
            if (0 > l) { w = w + l; l = 0; }
          }
          if (dir.indexOf('n') >= 0) {
            h = Math.max(minH, sh - dy); t = stp + sh - h;
            if (0 > t) { h = h + t; t = 0; }
          }
          win.style.left = l + 'px'; win.style.top = t + 'px';
          win.style.width = w + 'px'; win.style.height = h + 'px';
        }
        function up() {
          hdl.removeEventListener('pointermove', mv);
          hdl.removeEventListener('pointerup', up);
          hdl.removeEventListener('pointercancel', up);
          P.x = parseInt(win.style.left, 10); P.y = parseInt(win.style.top, 10);
          P.w = parseInt(win.style.width, 10); P.h = parseInt(win.style.height, 10);
          save();
        }
        hdl.addEventListener('pointermove', mv);
        hdl.addEventListener('pointerup', up);
        hdl.addEventListener('pointercancel', up);
        e.preventDefault(); e.stopPropagation();
      });
      win.appendChild(hdl);
    });
  }

  function initUI(S) {
    var KEY = 'simFlowPrefs_v1';
    var P = { open: true, x: null, y: null, w: 880, h: 520, z: 0.8, ruta: 1, etq: 1 };
    try {
      var raw = W.localStorage.getItem(KEY);
      if (raw) { var o = JSON.parse(raw); for (var k in o) { P[k] = o[k]; } }
    } catch (e) {}
    function save() { try { W.localStorage.setItem(KEY, JSON.stringify(P)); } catch (e) {} }
    S.P = P;

    var css = D.createElement('style');
    css.id = 'sim-flow-style';
    css.textContent =
      '#sim-flow-root{font:12px/1.45 Inter,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;}' +
      '#sim-flow-btn{position:fixed;right:118px;bottom:18px;z-index:1000001;background:#111827;color:#e5e7eb;' +
      'border:1px solid #2b3547;border-radius:20px;padding:7px 14px;cursor:pointer;display:flex;gap:8px;align-items:center;' +
      'font:600 12px Inter,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;box-shadow:0 2px 10px rgba(0,0,0,.35);}' +
      '#sim-flow-btn:hover{background:#1f2937;}' +
      '#sim-flow-btn .bdg{border-radius:9px;padding:0 7px;font-size:11px;color:#111;font-weight:700;}' +
      '#sim-flow-win{position:fixed;z-index:1000000;background:#0f172a;color:#e5e7eb;border:1px solid #334155;' +
      'border-radius:12px;box-shadow:0 8px 30px rgba(0,0,0,.45);display:flex;flex-direction:column;overflow:hidden;' +
      'min-width:420px;min-height:230px;box-sizing:border-box;}' +
      '.sim-rz{position:absolute;z-index:10;touch-action:none;}' +
      '.sim-rz-n{top:0;left:10px;right:10px;height:6px;cursor:ns-resize;}' +
      '.sim-rz-s{bottom:0;left:10px;right:10px;height:6px;cursor:ns-resize;}' +
      '.sim-rz-e{right:0;top:10px;bottom:10px;width:6px;cursor:ew-resize;}' +
      '.sim-rz-w{left:0;top:10px;bottom:10px;width:6px;cursor:ew-resize;}' +
      '.sim-rz-ne{top:0;right:0;width:12px;height:12px;cursor:nesw-resize;}' +
      '.sim-rz-nw{top:0;left:0;width:12px;height:12px;cursor:nwse-resize;}' +
      '.sim-rz-sw{bottom:0;left:0;width:12px;height:12px;cursor:nesw-resize;}' +
      '.sim-rz-se{bottom:0;right:0;width:16px;height:16px;cursor:nwse-resize;background:linear-gradient(135deg,rgba(0,0,0,0) 50%,#94a3b8 50%,#94a3b8 60%,rgba(0,0,0,0) 60%,rgba(0,0,0,0) 72%,#94a3b8 72%,#94a3b8 82%,rgba(0,0,0,0) 82%);}' +
      '#sim-flow-hd{display:flex;align-items:center;gap:6px;padding:6px 10px;background:#172033;cursor:move;' +
      'user-select:none;-webkit-user-select:none;border-bottom:1px solid #2b3547;flex:0 0 auto;touch-action:none;flex-wrap:wrap;}' +
      '#sim-flow-hd .ttl{font-weight:700;color:#fff;margin-right:auto;white-space:nowrap;}' +
      '#sim-flow-hd button{background:#243049;color:#e5e7eb;border:1px solid #334155;border-radius:4px;padding:1px 8px;' +
      'cursor:pointer;font:inherit;line-height:1.5;}' +
      '#sim-flow-hd button:hover{background:#334155;}' +
      '#sim-flow-hd button.chip{font-weight:700;opacity:.4;}' +
      '#sim-flow-hd button.chip.on{opacity:1;}' +
      '#sim-flow-bd{flex:1 1 auto;min-height:0;overflow:auto;background:#0b1220;}' +
      '#sim-flow-det{flex:0 0 auto;max-height:130px;overflow:auto;background:#111a2b;border-top:1px solid #2b3547;' +
      'padding:5px 10px;font-size:11px;}' +
      '#sim-flow-det .ev{display:flex;gap:10px;padding:1px 0;border-bottom:1px solid #1e293b;}' +
      '#sim-flow-det .ev .t{color:#94a3b8;white-space:nowrap;}' +
      '#sim-flow-det .ev .l{font-weight:700;white-space:nowrap;min-width:62px;}' +
      '#sim-flow-det .ev .m{white-space:pre-wrap;word-break:break-word;}' +
      '#sim-flow-ft{flex:0 0 auto;padding:3px 24px 3px 10px;background:#111a2b;color:#94a3b8;border-top:1px solid #2b3547;font-size:11px;' +
      'display:flex;gap:12px;flex-wrap:wrap;align-items:center;}' +
      '#sim-flow-ft .lg{display:inline-flex;align-items:center;gap:4px;}' +
      '#sim-flow-ft .lg i{display:inline-block;width:9px;height:9px;border-radius:2px;}' +
      '#sim-flow-bd .sf-nodo{cursor:pointer;}' +
      '#sim-flow-bd .sf-act .sf-caja{filter:drop-shadow(0 0 6px rgba(90,169,255,.75));}' +
      '#sim-flow-bd .sf-flow{stroke-dasharray:6 5;animation:sfflow 1s linear infinite;}' +
      '#sim-flow-bd .sf-flash .sf-caja{animation:sfpulse 1.6s ease-out 1;}' +
      '#sim-flow-bd .sf-live .sf-caja{stroke:#a855f7 !important;stroke-width:4;animation:sflive .9s ease-in-out infinite;}' +
      '@keyframes sflive{0%,100%{fill:#2a2340;filter:drop-shadow(0 0 3px rgba(168,85,247,.55));}50%{fill:#46307a;filter:drop-shadow(0 0 16px rgba(168,85,247,1));}}' +
      '@keyframes sfpulse{0%{filter:drop-shadow(0 0 0 rgba(240,171,252,0));}30%{filter:drop-shadow(0 0 12px rgba(240,171,252,.95));}100%{filter:drop-shadow(0 0 0 rgba(240,171,252,0));}}' +
      '@keyframes sfflow{to{stroke-dashoffset:-22;}}';
    D.head.appendChild(css);

    var root = D.createElement('div'); root.id = 'sim-flow-root';
    var win = D.createElement('div'); win.id = 'sim-flow-win';
    var hd = D.createElement('div'); hd.id = 'sim-flow-hd';
    var ttl = D.createElement('span'); ttl.className = 'ttl'; ttl.textContent = 'Flujo de datos';
    hd.appendChild(ttl);

    var chips = {};
    function mkChip(key, txt, title) {
      var b = D.createElement('button'); b.className = 'chip'; b.textContent = txt; b.title = title;
      b.addEventListener('click', function () { P[key] = P[key] ? 0 : 1; save(); S.render(); });
      chips[key] = b; hd.appendChild(b);
    }
    mkChip('etq', 'Etiquetas', 'Muestra/oculta las etiquetas de los conectores');
    function mkBtn(txt, title, fn) {
      var b = D.createElement('button'); b.textContent = txt; b.title = title;
      b.addEventListener('click', fn); hd.appendChild(b); return b;
    }
    mkBtn('-', 'Alejar', function () { P.z = Math.max(0.3, +(P.z - 0.1).toFixed(2)); save(); S.render(); });
    mkBtn('+', 'Acercar', function () { P.z = Math.min(1.6, +(P.z + 0.1).toFixed(2)); save(); S.render(); });
    mkBtn('Ajustar', 'Ajusta el diagrama al ancho de la ventana', function () {
      P.z = Math.max(0.3, Math.min(1.6, +((bd.clientWidth - 14) / VW).toFixed(2))); save(); S.render();
    });
    mkBtn('\u2715', 'Ocultar ventana (Ctrl + Alt + F)', function () { S.setOpen(false); });

    var bd = D.createElement('div'); bd.id = 'sim-flow-bd';
    var det = D.createElement('div'); det.id = 'sim-flow-det';
    var ft = D.createElement('div'); ft.id = 'sim-flow-ft';
    win.appendChild(hd); win.appendChild(bd); win.appendChild(det); win.appendChild(ft);

    var btn = D.createElement('div'); btn.id = 'sim-flow-btn'; btn.title = 'Mostrar/ocultar flujo de datos (Ctrl + Alt + F)';
    var btnTxt = D.createElement('span'); btnTxt.textContent = '\u25A3 Flujo';
    var badge = D.createElement('span'); badge.className = 'bdg';
    btn.appendChild(btnTxt); btn.appendChild(badge);
    btn.addEventListener('click', function () { S.setOpen(!P.open); });

    root.appendChild(win); root.appendChild(btn); D.body.appendChild(root);
    S.root = root;
    addResize(win, P, save, 420, 230);

    bd.addEventListener('click', function (e) {
      var g = e.target && e.target.closest ? e.target.closest('[data-nid]') : null;
      if (g) { S.sel = g.getAttribute('data-nid'); S.render(); }
    });
    // Doble clic en un nodo: la pagina se desplaza hasta ese menu.
    bd.addEventListener('dblclick', function (e) {
      try {
        var g = e.target && e.target.closest ? e.target.closest('[data-nid]') : null;
        if (!g) return;
        var nn = (S.data && S.data.nodos || {})[g.getAttribute('data-nid')];
        if (!nn || !nn.titulo) return;
        var hs = D.querySelectorAll('h3');
        for (var i = 0; i < hs.length; i++) {
          if ((hs[i].textContent || '').indexOf(nn.titulo) >= 0) { hs[i].scrollIntoView({ behavior: 'smooth', block: 'start' }); break; }
        }
      } catch (er) {}
    });

    function place() {
      var vw = W.innerWidth, vh = W.innerHeight;
      var w = Math.min(P.w, vw - 20), h = Math.min(P.h, vh - 80);
      win.style.width = w + 'px'; win.style.height = h + 'px';
      var x = (P.x === null || P.x === undefined) ? 18 : P.x;
      var y = (P.y === null || P.y === undefined) ? vh - h - 64 : P.y;
      x = Math.max(0, Math.min(x, vw - 80)); y = Math.max(0, Math.min(y, vh - 40));
      win.style.left = x + 'px'; win.style.top = y + 'px';
    }

    hd.addEventListener('pointerdown', function (e) {
      if (e.target && e.target.closest && e.target.closest('button')) return;
      var r = win.getBoundingClientRect(); var dx = e.clientX - r.left, dy = e.clientY - r.top;
      try { hd.setPointerCapture(e.pointerId); } catch (er) {}
      function mv(ev) {
        var x = Math.max(0, Math.min(ev.clientX - dx, W.innerWidth - 80));
        var y = Math.max(0, Math.min(ev.clientY - dy, W.innerHeight - 40));
        win.style.left = x + 'px'; win.style.top = y + 'px';
      }
      function up() {
        hd.removeEventListener('pointermove', mv); hd.removeEventListener('pointerup', up);
        P.x = parseInt(win.style.left, 10); P.y = parseInt(win.style.top, 10); save();
      }
      hd.addEventListener('pointermove', mv); hd.addEventListener('pointerup', up);
      e.preventDefault();
    });

    if (W.ResizeObserver) {
      new W.ResizeObserver(function () {
        if (win.offsetWidth > 0 && win.offsetHeight > 0 && P.open) { P.w = win.offsetWidth; P.h = win.offsetHeight; save(); }
      }).observe(win);
    }
    W.addEventListener('resize', function () { if (P.open) place(); });

    // La ventana en la que se hace clic queda por encima de la otra (logs / flujo).
    S.pd = function (e) {
      try {
        var t = e.target;
        var lw = D.getElementById('sim-log-win');
        if (win.contains(t)) { win.style.zIndex = 1000002; if (lw) lw.style.zIndex = 1000000; }
        else if (lw && lw.contains(t)) { lw.style.zIndex = 1000002; win.style.zIndex = 1000000; }
      } catch (er) {}
    };
    D.addEventListener('pointerdown', S.pd, true);

    S.kd = function (e) {
      if (e.ctrlKey && e.altKey && (e.key === 'f' || e.key === 'F' || e.code === 'KeyF')) { e.preventDefault(); S.setOpen(!P.open); }
    };
    D.addEventListener('keydown', S.kd);

    // ---- Edicion en vivo: detecta en la pagina QUE menu esta tocando el usuario (antes de que Streamlit recalcule) ----
    S.live = {};
    function menuDe(el) {
      try {
        var N = (S.data && S.data.nodos) || {};
        var ids = Object.keys(N);
        var a = el;
        while (a && a !== D.body && a.nodeType === 1) {
          var hs = a.querySelectorAll ? a.querySelectorAll('h3') : [];
          if (hs.length) {
            var hit = [];
            for (var i = 0; i < hs.length; i++) {
              var tx = (hs[i].textContent || '').trim(), best = null, bl = 0;
              ids.forEach(function (id) {
                var t = N[id].titulo;
                if (id !== 'SYS' && t && tx.indexOf(t) >= 0 && t.length > bl) { best = id; bl = t.length; }
              });
              if (best && hit.indexOf(best) < 0) hit.push(best);
            }
            if (hit.length === 1) return hit[0];
            if (hit.length > 1) return null;
          }
          a = a.parentElement;
        }
      } catch (e) {}
      return null;
    }
    var SEL_W = 'input,textarea,select,button,label,[role="combobox"],[role="listbox"],[role="option"],[role="slider"],[role="switch"],[role="checkbox"],[role="radio"],[data-baseweb],[data-testid="stDataFrame"],[data-testid="stDataEditor"]';
    function esTexto(el) {
      return !!(el && el.matches && el.matches('input[type="text"],input[type="number"],input:not([type]),textarea'));
    }
    function marcar(el, tipo) {
      try {
        if (!el || !el.closest || el.closest('#sim-flow-root') || el.closest('#sim-log-root')) return;
        if (!el.closest(SEL_W)) return;
        var id = menuDe(el);
        if (!id) return;
        var nuevo = !S.live[id];
        S.live[id] = (tipo !== 'click' && esTexto(el)) ? Infinity : Date.now() + 2500;
        if (nuevo) S.render();
      } catch (e) {}
    }
    S.lv = [
      ['focusin', function (e) { marcar(e.target, 'foco'); }, true],
      ['input', function (e) { marcar(e.target, 'input'); }, true],
      ['change', function (e) { marcar(e.target, 'click'); }, true],
      ['pointerdown', function (e) { marcar(e.target, 'click'); }, true],
      ['focusout', function (e) {
        try {
          var id = menuDe(e.target);
          if (id && S.live[id] === Infinity) S.live[id] = Date.now() + 700;
        } catch (er) {}
      }, true]
    ];
    S.lv.forEach(function (x) { D.addEventListener(x[0], x[1], x[2]); });
    S.lvT = setInterval(function () {
      var ids = Object.keys(S.live);
      if (!ids.length) return;
      var now = Date.now(), ch = false;
      ids.forEach(function (id) {
        if (S.live[id] === Infinity) {  // sigue "editando" mientras el foco este dentro de ese menu
          var ae = D.activeElement;
          if (!(ae && menuDe(ae) === id)) S.live[id] = now + 700;
        } else if (S.live[id] < now) { delete S.live[id]; ch = true; }
      });
      if (ch) S.render();
    }, 300);

    S.setOpen = function (v) {
      P.open = !!v; save();
      win.style.display = P.open ? 'flex' : 'none';
      if (P.open) { place(); S.render(); }
    };

    S.render = function () {
      var d = S.data || {}, N = d.nodos || {}, act = d.activo;
      // Nodos con actividad nueva desde el ultimo envio del servidor: parpadean una vez.
      if (S.flashSeq !== d.seq) {
        var fl = {}, mp = {};
        for (var kk in N) {
          mp[kk] = N[kk].seq;
          if (S.prevSeqMap && S.prevSeqMap[kk] !== N[kk].seq && N[kk].seq > 0) fl[kk] = 1;
        }
        S.flash = fl; S.prevSeqMap = mp; S.flashSeq = d.seq;
      }
      var FL = S.flash || {}; S.flash = {};
      function est(id) { return N[id] ? N[id].estado : 'pendiente'; }
      function arista(a, b) {
        var ea = est(a), eb = est(b);
        if (ea === 'error' || eb === 'error') return 'error';
        if (ea === 'omitido') return 'omitido';
        return ea;
      }
      function fluye(a, b, e) { return (a === act || b === act) && e !== 'pendiente' && e !== 'omitido'; }
      function estiloLinea(e, f) {
        return ' stroke="' + COL[e] + '" stroke-width="2"' + (e === 'omitido' ? ' stroke-dasharray="3 4"' : '') +
               (f ? ' class="sf-flow"' : '') + (e === 'pendiente' ? ' stroke-opacity="0.7"' : '');
      }

      var h = [];
      h.push('<svg xmlns="http://www.w3.org/2000/svg" width="' + (VW * P.z) + '" height="' + (VH * P.z) + '" viewBox="0 0 ' + VW + ' ' + VH + '" style="display:block;font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,sans-serif">');
      h.push('');

      // Grupos (cajas de color por etapa)
      GEO.forEach(function (o) {
        var g = o.g;
        h.push('<rect x="' + g.x + '" y="' + g.y + '" width="' + GW + '" height="' + o.h + '" rx="16" fill="' + g.c + '" fill-opacity="0.05" stroke="' + g.c + '" stroke-opacity="0.55" stroke-width="1.5"/>');
        h.push('<text x="' + (g.x + 16) + '" y="' + (g.y + 24) + '" fill="' + g.c + '" font-size="11" font-weight="700">' + esc(g.t) + '</text>');
      });

      // Conectores dentro de cada etapa (encadenados, de arriba hacia abajo)
      GRUPOS.forEach(function (g) {
        for (var i = 0; i < g.n.length - 1; i++) {
          var a = g.n[i], b = g.n[i + 1], pa = POS[a], pb = POS[b];
          var x = pa.x + NW / 2, y1 = pa.y + NH, y2 = pb.y, ym = (y1 + y2) / 2;
          var e = arista(a, b), f = fluye(a, b, e);
          h.push('<line x1="' + x + '" y1="' + y1 + '" x2="' + x + '" y2="' + y2 + '"' + estiloLinea(e, f) + '/>');
          h.push('<g transform="translate(' + x + ',' + ym + ')"><circle r="6.5" fill="' + COL[e] + '"/>' +
                 '<path d="M-2.6,-1.2 L0,1.6 L2.6,-1.2" fill="none" stroke="#0b1220" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></g>');
        }
      });

      // Conectores entre etapas (con etiqueta tipo pastilla)
      CRUCES.forEach(function (c) {
        var pa = POS[c[0]], pb = POS[c[1]];
        var x1 = pa.x + NW, y1 = pa.y + NH / 2, x2 = pb.x, y2 = pb.y + NH / 2, xm = x1 + 26;
        var e = arista(c[0], c[1]), f = fluye(c[0], c[1], e);
        h.push('<g><title>' + esc('M' + c[0] + ' -> M' + c[1] + ': ' + c[3]) + '</title>');
        h.push('<path d="M' + x1 + ' ' + y1 + ' H' + xm + ' V' + y2 + ' H' + x2 + '" fill="none" stroke-linejoin="round"' + estiloLinea(e, f) + '/>');
        if (P.etq) {
          var lw = c[2].length * 5.6 + 16, cx = (xm + x2) / 2 - 4;
          h.push('<rect x="' + (cx - lw / 2) + '" y="' + (y2 - 8) + '" width="' + lw + '" height="16" rx="8" fill="#eef0f6"/>');
          h.push('<text x="' + cx + '" y="' + (y2 + 3.5) + '" text-anchor="middle" fill="#1b1e27" font-size="9.5">' + esc(c[2]) + '</text>');
        }
        h.push('<g transform="translate(' + (x2 - 13) + ',' + y2 + ')"><circle r="6" fill="' + COL[e] + '"/>' +
               '<path d="M-1.4,-2.6 L1.6,0 L-1.4,2.6" fill="none" stroke="#0b1220" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></g></g>');
      });

      // Nodos (menus)
      var nErr = 0, nAdv = 0, nOmi = 0, nTot = 0, nEd = 0;
      GRUPOS.forEach(function (g) {
        g.n.forEach(function (id) {
          var p = POS[id], n = N[id], e = est(id), col = COL[e];
          if (n && n.seq > 0) { nTot++; }
          if (n && n.editado) { nEd++; }
          if (e === 'error') nErr++; else if (e === 'advertencia') nAdv++; else if (e === 'omitido') nOmi++;
          var etqNodo = id === 'SYS' ? 'Sistema' : 'M' + id + '  ' + (CORTO[id] || (n ? n.titulo : ''));
          var obs = !!(n && n.obsoleto);
          var editado = !!(n && n.editado);
          var pres = !!(n && n.presente);
          var vgl = (n && n.vig && n.vig.length) ? n.vig[n.vig.length - 1][1] : '';
          var sub = !pres ? (n && n.visto ? 'No disponible ahora' : 'No mostrado')
                          : (obs ? 'Desactualizado: volver a simular' : (vgl || (n && n.ultimo) || 'Menu mostrado'));
          var cps = [];
          if (n && n.ok) cps.push('ok ' + n.ok);
          if (n && n.warn) cps.push('adv ' + n.warn);
          if (n && n.err) cps.push('err ' + n.err);
          if (n && n.cambios) cps.push('cam ' + n.cambios);
          if (n && n.vistas) cps.push('vis ' + n.vistas);
          var cnt = cps.length ? cps.join(' ') : 'sin actividad';
          if (e === 'omitido') cnt = 'Valores por defecto';
          var tip = (n ? 'M' + id + ' - ' + n.titulo : etqNodo) + '\nEstado: ' + (editado ? 'Editado por el usuario + ' : '') + NOMBRE_ESTADO[e] + (n && n.tiempo !== null && n.tiempo !== undefined ? '\nTiempo: ' + fmtDur(n.tiempo) : '') + (n && n.ultimo ? '\nUltimo: ' + n.ultimo : '');
          h.push('<g class="sf-nodo' + (id === act ? ' sf-act' : '') + (FL[id] ? ' sf-flash' : '') + (S.live && S.live[id] ? ' sf-live' : '') + '" data-nid="' + esc(id) + '"' + (e === 'pendiente' ? ' opacity="0.8"' : '') + '><title>' + esc(tip) + '</title>');
          h.push('<rect class="sf-caja" x="' + p.x + '" y="' + p.y + '" width="' + NW + '" height="' + NH + '" rx="10" fill="#18233a" stroke="' + col +
                 '" stroke-width="' + (S.sel === id ? 3 : 1.5) + '"' + ((e === 'omitido' || obs) ? ' stroke-dasharray="5 3"' : '') + '/>');
          if (editado) { h.push('<rect x="' + (p.x + 2) + '" y="' + (p.y + 10) + '" width="3" height="' + (NH - 20) + '" rx="1.5" fill="' + COL_EDIT + '"/>'); }
          h.push(dot(p.x, p.y + NH / 2, col) + dot(p.x + NW, p.y + NH / 2, col));
          h.push('<text x="' + (p.x + 10) + '" y="' + (p.y + 18) + '" fill="#ffffff" font-size="11" font-weight="700">' + esc(cortar(etqNodo, 27)) + '</text>');
          var yEst = p.y + 59;
          var progTxt = (id === '16' && n && n.progreso !== null && n.progreso !== undefined) ? ' ' + n.progreso + '%' : '';
          var txtRes = (obs ? 'DESACTUALIZADO' : ETQ[e]) + progTxt;
          if (S.live && S.live[id]) {
            h.push('<text x="' + (p.x + 10) + '" y="' + yEst + '" fill="#a855f7" font-size="9" font-weight="700">EDITANDO AHORA...</text>');
          } else if (editado) {
            h.push('<text x="' + (p.x + 10) + '" y="' + yEst + '" font-size="9" font-weight="700"><tspan fill="' + COL_EDIT + '">EDITADO</tspan><tspan fill="#7d869c"> + </tspan><tspan fill="' + col + '">' + esc(txtRes) + '</tspan></text>');
          } else {
            h.push('<text x="' + (p.x + 10) + '" y="' + yEst + '" fill="' + col + '" font-size="9" font-weight="700">' + esc(txtRes) + '</text>');
          }
          if (n && n.tiempo !== null && n.tiempo !== undefined) {
            h.push('<text x="' + (p.x + NW - 10) + '" y="' + yEst + '" text-anchor="end" fill="#9aa3b8" font-size="9">' + esc((n.tiempo_tipo === 'sim' ? 'sim ' : '') + fmtDur(n.tiempo)) + '</text>');
          }
          h.push('<text x="' + (p.x + 10) + '" y="' + (p.y + 32) + '" fill="#9aa3b8" font-size="9.5">' + esc(cortar(sub, 31)) + '</text>');
          h.push('<text x="' + (p.x + 10) + '" y="' + (p.y + 45) + '" fill="#7d869c" font-size="9">' + esc(cortar(cnt, 33)) + '</text>');
          if (id === '16' && n && n.progreso !== null && n.progreso !== undefined && n.progreso < 100) {
            h.push('<rect x="' + (p.x + 8) + '" y="' + (p.y + NH - 4) + '" width="' + ((NW - 16) * n.progreso / 100) + '" height="2.5" rx="1" fill="#5aa9ff"/>');
          }
          h.push('</g>');
        });
      });
      h.push('</svg>');

      var sl = bd.scrollLeft, st = bd.scrollTop;
      bd.innerHTML = h.join('');
      bd.scrollLeft = sl; bd.scrollTop = st;

      // Detalle del nodo elegido (o del ultimo activo)
      var sel = S.sel || act, dh = '';
      if (sel && POS[sel]) {
        var ns = N[sel], es = est(sel);
        var edS = !!(ns && ns.editado);
        dh += '<div style="font-weight:700;margin-bottom:3px;color:#fff">' + esc(sel === 'SYS' ? 'Sistema' : 'M' + sel + ' - ' + (ns ? ns.titulo : CORTO[sel])) +
              ' <span style="color:' + COL[es] + '">[' + (edS ? '<span style="color:' + COL_EDIT + '">Editado por el usuario</span> + ' : '') + esc(NOMBRE_ESTADO[es]) + ']</span></div>';
        var partesT = [];
        if (ns && ns.dur_proc !== null && ns.dur_proc !== undefined) partesT.push('simulacion completa: ' + fmtDur(ns.dur_proc));
        if (ns && ns.dur_accion !== null && ns.dur_accion !== undefined) partesT.push('procesar el ultimo cambio: ' + fmtDur(ns.dur_accion));
        if (ns && ns.dur !== null && ns.dur !== undefined) partesT.push('ultima carga del menu: ' + fmtDur(ns.dur));
        if (partesT.length) dh += '<div style="color:#9aa3b8;margin-bottom:3px">Tiempos - ' + esc(partesT.join(' | ')) + '</div>';
        if (ns && ns.eventos && ns.eventos.length) {
          ns.eventos.slice().reverse().forEach(function (ev) {
            dh += '<div class="ev"><span class="t">' + esc(ev[0]) + '</span><span class="l" style="color:' + (COL_NIVEL[ev[1]] || '#e5e7eb') + '">' + esc(ev[1]) +
                  '</span><span class="m">' + esc(ev[2]) + '</span></div>';
          });
        } else {
          dh += '<div style="opacity:.6">' + (ns && ns.visto ? 'Menu mostrado, sin eventos registrados todavia.' : 'Este menu aun no se muestra o no registra eventos.') + '</div>';
        }
      } else {
        dh = '<div style="opacity:.6">Sin actividad todavia. Haz clic en un nodo para ver su detalle.</div>';
      }
      det.innerHTML = dh;

      for (var k in chips) { chips[k].className = 'chip' + (P[k] ? ' on' : ''); }
      var lg = '';
      ['exito', 'advertencia', 'error', 'omitido', 'activo', 'pendiente'].forEach(function (s) {
        lg += '<span class="lg"><i style="background:' + COL[s] + '"></i>' + NOMBRE_ESTADO[s] + '</span>';
      });
      lg += '<span class="lg"><i style="background:#a855f7"></i>Editando ahora</span>';
      lg += '<span class="lg"><i style="background:' + COL_EDIT + '"></i>Editado por el usuario (se combina: EDITADO + EXITOSO / ADVERTENCIA / ERROR)</span>';
      ft.innerHTML = lg + '<span style="margin-left:auto">' + nTot + ' menus con actividad | ' + nErr + ' err | ' + nAdv + ' adv | ' + nOmi + ' omitidos | ' + nEd + ' editados | zoom ' + Math.round(P.z * 100) + '%</span>';
      badge.textContent = nErr ? (nErr + ' err') : (nAdv ? (nAdv + ' adv') : (nOmi ? (nOmi + ' omit') : String(nTot)));
      badge.style.background = nErr ? '#ff7b72' : (nAdv ? '#e3b341' : (nOmi ? '#9aa3b8' : '#7aa2f7'));
    };

    place();
    win.style.display = P.open ? 'flex' : 'none';
  }

  var S = W.__simFlow;
  if (!S || S.ver !== VER) {
    if (S) {
      try { if (S.root && S.root.parentNode) S.root.parentNode.removeChild(S.root); } catch (e) {}
      try { if (S.kd) D.removeEventListener('keydown', S.kd); } catch (e) {}
      try { if (S.pd) D.removeEventListener('pointerdown', S.pd, true); } catch (e) {}
      try { (S.lv || []).forEach(function (x) { D.removeEventListener(x[0], x[1], x[2]); }); clearInterval(S.lvT); } catch (e) {}
      try { var oldCss = D.getElementById('sim-flow-style'); if (oldCss) oldCss.parentNode.removeChild(oldCss); } catch (e) {}
    }
    S = W.__simFlow = { ver: VER, data: {}, sel: null };
    initUI(S);
  }
  S.data = DATA;
  S.render();
})();
</script>
'''


def _flujo_sanear_js(plantilla):
    """Streamlit sanea el HTML con DOMPurify, que ELIMINA cualquier <script> cuyo texto contenga algo con
    forma de etiqueta (un '<' seguido de letra, '/' o '!'), p. ej. '<svg' o '</text>' dentro de una cadena JS.
    Dentro del script se reemplaza cada uno de esos '<' por su escape \\u003c (identico en ejecucion, pero
    sin el caracter '<' literal), igual que la plantilla de logs que evita '<' por completo."""
    ini = plantilla.index('<script>') + len('<script>')
    fin = plantilla.rindex('</script>')
    cuerpo = re.sub(r'<(?=[/\w!])', lambda _m: '\\u003c', plantilla[ini:fin])
    return plantilla[:ini] + cuerpo + plantilla[fin:]


_FLUJO_PLANTILLA_HTML = _flujo_sanear_js(_FLUJO_PLANTILLA_HTML)


def _flujo_refrescar(force=False):
    """Redibuja la ventana flotante de flujo de datos. Se llama junto con _logs_refrescar, asi que
    tambien se actualiza en vivo durante la simulacion (limitada a ~2 veces por segundo)."""
    try:
        slot = getattr(_LOGS, 'slot_flujo', None)
        if slot is None:
            return
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is None:  # hilo de trabajo: Streamlit no permite dibujar desde ahi
            return
        ahora = time.time()
        if not force and ahora - getattr(_LOGS, 'ultimo_refresco_flujo', 0.0) < 0.5:
            return
        _LOGS.ultimo_refresco_flujo = ahora
        with _LOGS.lock:
            f = _flujo_estado()
            datos_py = {
                'nodos': _flujo_vista_nodos(f),
                'ruta': [list(r) for r in f['ruta'][-40:]],
                'activo': f.get('activo'),
                'seq': f.get('seq', 0),
            }
        datos = (json.dumps(datos_py, ensure_ascii=False).replace('<', '\\u003c')
                 .replace('\u2028', '\\u2028').replace('\u2029', '\\u2029'))
        html = _FLUJO_PLANTILLA_HTML.replace('__VER__', 'f6').replace('__DATA__', datos)
        try:
            slot.html(html, unsafe_allow_javascript=True)
        except TypeError:  # Streamlit antiguo sin unsafe_allow_javascript: se usa un iframe sin altura
            import streamlit.components.v1 as components
            with slot:
                components.html(html, height=0)
    except Exception:
        pass


_logs_instalar_parches()
_logs_nuevo_run()
_LOGS.log = log_evento
_LOGS.slot = st.empty()  # aqui vive (invisible) la ventana flotante de logs
_LOGS.slot_flujo = st.empty()  # aqui vive (invisible) la ventana flotante de flujo de datos
if not st.session_state.get('_log_sesion_iniciada'):
    st.session_state['_log_sesion_iniciada'] = True
    # Sesion nueva (recarga de la pagina o servidor reiniciado): logs y flujo empiezan desde cero.
    with _LOGS.lock:
        _LOGS.items.clear()
        _LOGS.prev = set()
        _LOGS.cur = set()
        _LOGS.seq = 0
        _LOGS.flujo = None
    log_evento('INFO', 'Sesión iniciada: simulador de túneles listo.')
_logs_refrescar(force=True)  # la ventana aparece desde el principio, antes de cargar nada


# ======================================================================
# #####MOTOR DE SIMULACIÓN Y FUNCIONES AUXILIARES (compartidas por varios menús)#####
# ======================================================================
# ----------------------------------------------------
# Tablas centradas y ordenadas (envoltorio de st.dataframe)
# ----------------------------------------------------
# Streamlit por defecto deja el texto alineado a la izquierda y los numeros
# a la derecha, con lo que las tablas se ven descuadradas entre encabezado
# y valores. Esta funcion reemplaza a st.dataframe en toda la app: centra
# automaticamente TODAS las columnas (texto y numeros, encabezado incluido)
# usando column_config, sin tener que configurar cada tabla a mano.
def _arrow_seguro(df):
    """Devuelve el DataFrame con las columnas de tipos mezclados (ej. numeros y texto en una
    misma columna 'object') convertidas a texto, para que Streamlit pueda pasarlo a Arrow sin
    imprimir el aviso 'Serialization of dataframe to Arrow table was unsuccessful'."""
    import numbers
    if not isinstance(df, pd.DataFrame):
        return df
    salida = df
    for col in df.columns:
        serie = df[col]
        if isinstance(serie, pd.DataFrame) or serie.dtype != object:
            continue
        tipos = {type(v) for v in serie.dropna()}
        if len(tipos) <= 1:
            continue
        if all(issubclass(t, numbers.Number) for t in tipos):
            continue  # int + float mezclados: Arrow lo maneja bien
        if salida is df:
            salida = df.copy()
        salida[col] = serie.map(lambda v: v if pd.isna(v) is True else str(v))
    return salida


def _formato_celda_defecto(v):
    """Formato por defecto de las tablas HTML: numeros con hasta 4 decimales sin ceros sobrantes."""
    if isinstance(v, (float, np.floating)):
        if pd.isna(v):
            return ""
        return f"{v:.4f}".rstrip('0').rstrip('.') if np.isfinite(v) else str(v)
    return v


_MAX_FILAS_TABLA_HTML = 3000


def _tabla_html_centrada(data, hide_index, altura):
    """Dibuja un DataFrame o Styler como tabla HTML con encabezado Y celdas centrados.
    (st.dataframe usa un canvas: su encabezado no se puede centrar ni con column_config ni con CSS.)
    Devuelve True si pudo dibujarla."""
    sty = data if hasattr(data, 'to_html') and hasattr(data, 'data') else data.style
    df = sty.data
    if hide_index:
        sty = sty.hide(axis='index')
    # Formato por defecto solo en las columnas que el llamador no formateo explicitamente.
    ya_formateadas = {c for (_, c) in getattr(sty, '_display_funcs', {}).keys()}
    for j, col in enumerate(df.columns):
        if j not in ya_formateadas:
            sty = sty.format(_formato_celda_defecto, subset=[col], na_rep="", escape="html")
    sty = sty.set_table_attributes('class="tabla-centrada"')
    html = sty.to_html()
    if isinstance(altura, (int, float)) and altura > 0:
        estilo = f"max-height:{int(altura)}px;"
    else:
        estilo = "max-height:480px;"
    html = f'<div class="tabla-centrada-wrap" style="{estilo}">{html}</div>'
    if hasattr(st, 'html'):
        st.html(html)
    else:
        st.markdown(html.replace("\n", ""), unsafe_allow_html=True)
    return True


def mostrar_tabla(data, **kwargs):
    """Muestra una tabla de solo lectura con encabezado y valores CENTRADOS.

    Acepta los mismos argumentos que st.dataframe (incluido un Styler de pandas). Se dibuja como
    tabla HTML (encabezado centrado, fila de titulos fija al hacer scroll). Si la tabla es muy grande
    (> _MAX_FILAS_TABLA_HTML filas), usa column_config o necesita seleccion, vuelve a st.dataframe
    con las celdas centradas (ahi Streamlit no permite centrar el encabezado)."""
    try:
        df_base = data.data if hasattr(data, 'data') and isinstance(getattr(data, 'data', None), pd.DataFrame) else data
        columnas = list(df_base.columns)
    except Exception:
        columnas = list(data.columns)
        df_base = data

    column_config_usuario = kwargs.pop('column_config', None) or {}
    if 'use_container_width' in kwargs:   # obsoleto en Streamlit: se traduce a 'width'
        kwargs['width'] = 'stretch' if kwargs.pop('use_container_width') else 'content'
    kwargs.setdefault('width', 'stretch')
    kwargs.setdefault('hide_index', True)

    usable_html = (
        isinstance(df_base, pd.DataFrame)
        and not column_config_usuario
        and not kwargs.get('on_select') and not kwargs.get('selection_mode')
        and len(df_base) <= _MAX_FILAS_TABLA_HTML
        and df_base.columns.is_unique
    )
    if usable_html:
        try:
            if isinstance(data, pd.DataFrame):
                data = _arrow_seguro(data)
            _tabla_html_centrada(data, kwargs.get('hide_index', True), kwargs.get('height'))
            return None
        except Exception:
            pass   # cualquier problema de render -> vuelve a st.dataframe

    column_config_final = {
        col: st.column_config.Column(alignment="center")
        for col in columnas
        if col not in column_config_usuario
    }
    column_config_final.update(column_config_usuario)
    try:
        if isinstance(data, pd.DataFrame):
            data = _arrow_seguro(data)
        elif hasattr(data, 'data') and isinstance(data.data, pd.DataFrame):
            data.data = _arrow_seguro(data.data)  # Styler
    except Exception:
        pass
    return st.dataframe(data, column_config=column_config_final, **kwargs)


# CSS de respaldo: centra tambien el encabezado y despareja menos el alto
# de fila entre tablas (por si la version de Streamlit no soporta aun
# 'alignment' en column_config para algun tipo de columna).
st.markdown(
    """
    <style>
    [data-testid="stDataFrame"] div[data-testid="stDataFrameResizable"] {
        margin-left: auto;
        margin-right: auto;
    }
    [data-testid="stDataFrame"] table th,
    [data-testid="stDataEditor"] table th {
        text-align: center !important;
    }
    [data-testid="stDataFrame"] table td,
    [data-testid="stDataEditor"] table td {
        text-align: center !important;
    }
    [data-testid="stDataEditor"] div[data-testid="stDataFrameResizable"] {
        margin-left: auto;
        margin-right: auto;
    }
    /* Tablas HTML de solo lectura (mostrar_tabla): encabezado y celdas centrados */
    .tabla-centrada-wrap {
        overflow: auto; width: 100%; margin: 0.25rem 0 0.75rem 0;
        border: 1px solid rgba(150,150,150,0.35); border-radius: 8px;
    }
    table.tabla-centrada {
        width: 100%; border-collapse: collapse; font-size: 0.9rem; table-layout: auto;
    }
    table.tabla-centrada th, table.tabla-centrada td {
        text-align: center !important; vertical-align: middle !important;
        padding: 0.45rem 0.75rem; border-bottom: 1px solid rgba(150,150,150,0.25);
        border-right: 1px solid rgba(150,150,150,0.15); white-space: nowrap;
    }
    table.tabla-centrada th:last-child, table.tabla-centrada td:last-child { border-right: none; }
    table.tabla-centrada thead th {
        position: sticky; top: 0; z-index: 1; font-weight: 600;
        background: rgba(150,150,150,0.16); backdrop-filter: blur(6px);
    }
    table.tabla-centrada tbody tr:last-child td { border-bottom: none; }
    table.tabla-centrada tbody tr:hover td { background: rgba(150,150,150,0.08); }
    </style>
    """,
    unsafe_allow_html=True,
)

_mostrar_tabla_original = mostrar_tabla


@functools.wraps(_mostrar_tabla_original)
def mostrar_tabla(data, **kwargs):
    """Igual que antes, pero deja registrada en logs/flujo la tabla que el usuario esta viendo."""
    _registrar_vista('tabla', data)
    _LOGS.depth += 1  # st.dataframe interno no se vuelve a registrar
    try:
        return _mostrar_tabla_original(data, **kwargs)
    finally:
        _LOGS.depth -= 1


# ----------------------------------------------------
# Formato unificado de menús (tarjetas)
# ----------------------------------------------------
# Todos los menús (1 a 16) se dibujan igual: una tarjeta con borde y un encabezado
# "[ícono] Título". Sin barra lateral ni menú hamburguesa: todo sigue en una sola página,
# de arriba hacia abajo. Los sub-títulos internos usan siempre el mismo nivel.
# Ícono Material de cada menú (el número ya no se muestra: es solo el identificador interno).
_ICONOS_MENU = {
    '1': 'upload_file', '2': 'manage_search', '3': 'inventory_2', '4': 'layers',
    '5': 'tune', '6': 'settings', '7': 'manage_history', '8': 'bar_chart',
    '9': 'add_task', '10': 'landscape', '11': 'error', '12': 'handyman',
    '13': 'checklist', '14': 'block', '15': 'view_in_ar', '16': 'play_arrow',
    '17': 'analytics', '18': 'timeline', '19': 'event_available', '20': 'report',
    '21': 'receipt_long', '22': 'download', '17b': 'precision_manufacturing', '20b': 'terrain',
}


@contextmanager
def seccion_menu(numero, titulo, ayuda=None):
    """Tarjeta estándar de un menú: borde + encabezado '[ícono] Título'. Se usa con `with`."""
    previo_menu = getattr(_LOGS, 'menu', None)
    _t0_menu = time.perf_counter()
    _LOGS.menu = f"M{numero} · {titulo}"  # los logs que ocurran dentro llevan esta etiqueta
    try:
        vistos = st.session_state.setdefault('_log_menus_vistos', set())
        nuevo = _flujo_menu_render(numero, titulo)  # marca el menu como dibujado en ESTA corrida
        if nuevo or str(numero) not in vistos:  # primera vez o reaparece (p. ej. tras quitar y volver a cargar un archivo)
            vistos.add(str(numero))
            log_evento('INFO', 'Sección mostrada')
        with st.container(border=True):
            icono = _ICONOS_MENU.get(str(numero))
            st.subheader(f":material/{icono}: {titulo}" if icono else titulo, help=ayuda)
            yield
    finally:
        _flujo_menu_duracion(numero, titulo, time.perf_counter() - _t0_menu)
        _LOGS.menu = previo_menu


def aplicar_secuencias_personalizadas(df_base, actividades_adicionales):
    """Decide la 'Secuencia' (posición en el orden de ejecución) de las actividades
    personalizadas y corre las existentes lo necesario para hacerles espacio.

    Cada actividad personalizada trae 'Posicion':
      - None / vacío -> va al final (queda después de la última secuencia existente).
      - N (entero)   -> queda EXACTAMENTE como secuencia N del listado final; las
                        actividades que estaban en N o más allá se corren +1.
    Las posiciones explícitas se aplican de menor a mayor, así que 'N' siempre es la
    posición en el listado final (ej. con 30 actividades y dos nuevas en 16 y 22,
    las nuevas quedan en 16 y 22 de un total de 32).

    Devuelve (df_base con 'Secuencia' ya corrida, lista de dicts de las personalizadas
    con su 'Secuencia' definitiva y sin la clave 'Posicion')."""
    df_base = df_base.copy()
    if not actividades_adicionales:
        return df_base, []
    seqs = pd.to_numeric(df_base['Secuencia'], errors='coerce')
    colocadas = []  # [dict, secuencia]

    def _maxima():
        m = seqs.max()
        m = 0 if pd.isna(m) else int(m)
        return max([m] + [s for _, s in colocadas])

    explicitas = sorted(
        [a for a in actividades_adicionales if a.get('Posicion')],
        key=lambda a: int(a['Posicion']))
    al_final = [a for a in actividades_adicionales if not a.get('Posicion')]
    for a in explicitas:
        p = min(max(int(a['Posicion']), 1), _maxima() + 1)
        seqs = seqs.where(~(seqs >= p), seqs + 1)
        for par in colocadas:
            if par[1] >= p:
                par[1] += 1
        colocadas.append([a, p])
    for a in al_final:
        colocadas.append([a, _maxima() + 1])

    if not seqs.isna().any():
        try:
            seqs = seqs.astype(df_base['Secuencia'].dtype)
        except (TypeError, ValueError):
            pass
    df_base['Secuencia'] = seqs
    nuevas = []
    for a, s in colocadas:
        d = {k: v for k, v in a.items() if k != 'Posicion'}
        d['Secuencia'] = int(s)
        nuevas.append(d)
    return df_base, nuevas


def subtitulo_menu(texto, ayuda=None):
    """Sub-título interno estándar dentro de la tarjeta de un menú. `ayuda` se muestra en el tooltip '?'."""
    st.markdown(f"#### {texto}", help=ayuda)


def ayuda_inline(texto, etiqueta="Instrucciones"):
    """Reemplazo de las narraciones largas: una etiqueta corta con el icono '?' que muestra la instrucción."""
    st.caption(etiqueta, help=texto)


# ----------------------------------------------------
# Reinicio automático de resultados cuando cambia la configuración
# ----------------------------------------------------
# Claves que se llenan al terminar una simulación (Menu 16) y que quedan
# obsoletas/inconsistentes si el usuario cambia algo DESPUES de simular
# (ej. corrió 200 simulaciones y luego bajó a 100, o modificó recursos,
# restricciones, actividades, etc.). En vez de dejar en pantalla resultados
# que ya no corresponden a la configuración actual, se limpian solas y se
# avisa con un st.warning para que el usuario sepa que debe volver a
# presionar "SIMULAR".
_CLAVES_RESULTADOS_SIMULACION = [
    'resultados', 'resultados_base_sin_cambios_flota', 'avance_por_turno_real',
    'estadisticas_recursos', 'traza_eventos', 'df_traza_eventos_cache', 'run_id',
    'ventanas_flota', 'registro_cambios_flota', 'nivel_simulado',
    'duracion_turno_simulada', 'numero_turnos_simulado', 'n_simulaciones_ejecutadas',
    'df_actividades_simuladas', 'df_frentes_nivel_simuladas',
    'bloqueos_tuneles_dict_simulado', 'tiempo_limite_simulado',
    'mwd_log_geologia', 'mwd_config_usada', 'mwd_catalogo_usado',
]


def reiniciar_resultados_simulacion(motivo: str):
    """Invalida los resultados de la última simulación corrida porque el usuario
    cambió algo que los deja desactualizados (ej. cambió el N° de Simulaciones
    Monte Carlo de 200 a 100 después de haber simulado, o modificó recursos,
    restricciones, actividades, etc.). No existe forma de "interrumpir" un bucle
    de simulación ya en curso dentro de un mismo click de "SIMULAR" (Streamlit
    corre ese bucle de una sola vez, de principio a fin, antes de volver a leer
    la pantalla), pero si el usuario cambia algo DESPUES de que una simulación
    ya terminó, esta función limpia esos resultados viejos para que no queden
    mezclados con la nueva configuración, y deja todo listo para que la
    siguiente simulación tome el cambio de inmediato al presionar "SIMULAR".
    No hace nada (ni muestra nada) si todavía no se ha corrido ninguna
    simulación, para no generar avisos innecesarios."""
    habia_resultados = 'resultados' in st.session_state
    if not habia_resultados:
        return
    for clave in _CLAVES_RESULTADOS_SIMULACION:
        st.session_state.pop(clave, None)
    st.warning(
        f"Simulación reiniciada: {motivo}. Los resultados anteriores quedaron "
        "obsoletos y se limpiaron para no mezclarlos con la nueva configuración. "
        "Presiona **SIMULAR** de nuevo para generar resultados actualizados."
    )


SHEETS_OPERACION = {
    'Actividades': ['Nivel', 'Secuencia', 'Actividad', 'Recurso', 'Tiempo', 'Distribucion'],
    'Recursos': ['Nivel', 'Recurso', 'Cantidad', 'Cambio', 'Turno_Cambio', 'Cantidad_Cambio'],
    'Frentes': ['Nivel', 'Index', 'Frentes', 'Xi', 'Yi', 'Xf', 'Yf']
}
SHEETS_RESTRICCIONES = {
    'FrenteHundimiento': ['X', 'Y'],
    'ObrasCiviles': ['X', 'Y'],
    'Restriccion': ['Tipo', 'X', 'Y', 'Turno']
}
DEFAULT_RESTRICTION_RADII = {
    'Polvorazo': 80.0,
    'Polvorazo B': 80.0,
    'PA FH': 40.0
}
VALID_DISTRIBUTIONS = {'Cte', 'fisk', 'normal', 'weibull', 'gamma', 'lognormal', 'kstwobign',
                       'rayleigh', 'foldcauchy', 'foldnorm', 'ncx2',
                       'burr', 'loglaplace', 'maxwell', 'nakagami'}

# ---- Parametros de cada distribucion, tal como los define scipy.stats ----
# Cada entrada describe los argumentos que scipy.stats.<dist>.pdf()/.rvs() usa
# realmente (nombre del parametro tal cual la libreria y su significado).
# Es la fuente unica para armar los formularios de la UI y el grafico de PDF.
DISTRIBUTION_PARAM_INFO = {
    'Cte': {
        'scipy_name': None,
        'parametros': [
            {'key': 'valor', 'label': 'Valor constante', 'min': 0.0001, 'default': 1.0},
        ],
    },
    'normal': {
        'scipy_name': 'scipy.stats.norm',
        'parametros': [
            {'key': 'mean', 'label': 'mean (promedio, mu)', 'default': 1.0},
            {'key': 'std', 'label': 'std (desviacion estandar, sigma)', 'min': 0.0001, 'default': 0.1},
        ],
    },
    'weibull': {
        'scipy_name': 'scipy.stats.weibull_min',
        'parametros': [
            {'key': 'c', 'label': 'c (shape, parametro de forma k)', 'min': 0.0001, 'default': 2.0},
            {'key': 'loc', 'label': 'loc (desplazamiento)', 'default': 0.0},
            {'key': 'scale', 'label': 'scale (parametro de escala lambda)', 'min': 0.0001, 'default': 1.0},
        ],
    },
    'gamma': {
        'scipy_name': 'scipy.stats.gamma',
        'parametros': [
            {'key': 'a', 'label': 'a (shape, parametro de forma k)', 'min': 0.0001, 'default': 2.0},
            {'key': 'loc', 'label': 'loc (desplazamiento)', 'default': 0.0},
            {'key': 'scale', 'label': 'scale (theta, escala; theta = 1/rate)', 'min': 0.0001, 'default': 0.5},
        ],
    },
    'lognormal': {
        'scipy_name': 'scipy.stats.lognorm',
        'parametros': [
            {'key': 's', 'label': 's (sigma, desv. estandar en escala log)', 'min': 0.0001, 'default': 0.5},
            {'key': 'loc', 'label': 'loc (desplazamiento)', 'default': 0.0},
            {'key': 'scale', 'label': 'scale (= exp(mu), mu en escala log)', 'min': 0.0001, 'default': 1.0},
        ],
    },
    'fisk': {
        'scipy_name': 'scipy.stats.fisk',
        'parametros': [
            {'key': 'c', 'label': 'c (shape, parametro de forma)', 'min': 0.0001, 'default': 6.26},
            {'key': 'loc', 'label': 'loc (desplazamiento)', 'default': -0.42},
            {'key': 'scale', 'label': 'scale (parametro de escala)', 'min': 0.0001, 'default': 3.92},
        ],
    },
    'kstwobign': {
        'scipy_name': 'scipy.stats.kstwobign',
        'parametros': [
            {'key': 'loc', 'label': 'loc (desplazamiento)', 'default': -0.48},
            {'key': 'scale', 'label': 'scale (parametro de escala)', 'min': 0.0001, 'default': 2.62},
        ],
    },
    'rayleigh': {
        'scipy_name': 'scipy.stats.rayleigh',
        'parametros': [
            {'key': 'loc', 'label': 'loc (desplazamiento)', 'default': 0.0},
            {'key': 'scale', 'label': 'scale (sigma, parametro de escala/modo)', 'min': 0.0001, 'default': 1.0},
        ],
    },
    'foldcauchy': {
        'scipy_name': 'scipy.stats.foldcauchy',
        'parametros': [
            {'key': 'c', 'label': 'c (shape, ubicacion de la Cauchy antes de plegar)', 'min': 0.0, 'default': 2.20},
            {'key': 'loc', 'label': 'loc (desplazamiento)', 'default': 0.0},
            {'key': 'scale', 'label': 'scale (parametro de escala)', 'min': 0.0001, 'default': 1.0},
        ],
    },
    'foldnorm': {
        'scipy_name': 'scipy.stats.foldnorm',
        'parametros': [
            {'key': 'c', 'label': 'c (shape, promedio de la normal antes de plegar, en unidades de scale)', 'min': 0.0, 'default': 1.79},
            {'key': 'loc', 'label': 'loc (desplazamiento)', 'default': 0.0},
            {'key': 'scale', 'label': 'scale (sigma, desviacion estandar antes de plegar)', 'min': 0.0001, 'default': 1.19},
        ],
    },
    'ncx2': {
        'scipy_name': 'scipy.stats.ncx2',
        'parametros': [
            {'key': 'df', 'label': 'df (grados de libertad)', 'min': 0.0001, 'default': 2.0},
            {'key': 'nc', 'label': 'nc (parametro de no-centralidad)', 'min': 0.0, 'default': 0.0},
            {'key': 'loc', 'label': 'loc (desplazamiento)', 'default': 0.0},
            {'key': 'scale', 'label': 'scale (parametro de escala)', 'min': 0.0001, 'default': 1.0},
        ],
    },
    'burr': {
        'scipy_name': 'scipy.stats.burr',
        'parametros': [
            {'key': 'c', 'label': 'c (shape, primer parametro de forma)', 'min': 0.0001, 'default': 4.0},
            {'key': 'd', 'label': 'd (shape, segundo parametro de forma)', 'min': 0.0001, 'default': 2.0},
            {'key': 'loc', 'label': 'loc (desplazamiento)', 'default': 0.0},
            {'key': 'scale', 'label': 'scale (parametro de escala)', 'min': 0.0001, 'default': 1.0},
        ],
    },
    'loglaplace': {
        'scipy_name': 'scipy.stats.loglaplace',
        'parametros': [
            {'key': 'c', 'label': 'c (shape, parametro de forma)', 'min': 0.0001, 'default': 2.0},
            {'key': 'loc', 'label': 'loc (desplazamiento)', 'default': 0.0},
            {'key': 'scale', 'label': 'scale (parametro de escala)', 'min': 0.0001, 'default': 1.0},
        ],
    },
    'maxwell': {
        'scipy_name': 'scipy.stats.maxwell',
        'parametros': [
            {'key': 'loc', 'label': 'loc (desplazamiento)', 'default': 0.0},
            {'key': 'scale', 'label': 'scale (a, parametro de escala)', 'min': 0.0001, 'default': 1.0},
        ],
    },
    'nakagami': {
        'scipy_name': 'scipy.stats.nakagami',
        'parametros': [
            {'key': 'nu', 'label': 'nu (shape, parametro de forma m)', 'min': 0.0001, 'default': 1.0},
            {'key': 'loc', 'label': 'loc (desplazamiento)', 'default': 0.0},
            {'key': 'scale', 'label': 'scale (parametro de escala/spread)', 'min': 0.0001, 'default': 1.0},
        ],
    },
}

SHEETS_MARINA = {
    'Vaciaderos': ['Nivel', 'Capacidad', 'Maximo', 'Tipo']
}
SHEETS_MUCKPILE = {
    'Muckpile': ['Nivel', 'Capacidad Ocupada', 'Maximo', 'Tipo']
}

# ---- Configuración del orquestador de tareas (Redis) ----
# En Streamlit Cloud no hay Redis local: la app corre sin orquestador salvo que se defina REDIS_URL
# (o REDIS_HOST) en Settings > Secrets, por ejemplo con un Redis gestionado (Upstash, Redis Cloud).
REDIS_URL = _cfg_nube("REDIS_URL")
REDIS_HOST = _cfg_nube("REDIS_HOST", "localhost")
REDIS_PORT = int(_cfg_nube("REDIS_PORT", 6379))
REDIS_DB = int(_cfg_nube("REDIS_DB", 0))
REDIS_PASSWORD = _cfg_nube("REDIS_PASSWORD")
REDIS_SSL = str(_cfg_nube("REDIS_SSL", "")).strip().lower() in ("1", "true", "si", "sí")
REDIS_LOCK_KEY = "orquestador:simulacion:lock"
REDIS_ESTADO_KEY = "orquestador:simulacion:estado"
REDIS_HISTORIAL_KEY = "orquestador:simulacion:historial"
REDIS_LOCK_TTL = int(os.environ.get("REDIS_LOCK_TTL", 1800))  # seg. TTL de seguridad del lock

@st.cache_resource(show_spinner=False)
def get_redis_client():
    """Conecta con Redis (orquestador de tareas). Si no está disponible, la app
    sigue funcionando sin orquestación (modo degradado) en vez de romperse."""
    if ES_NUBE and not REDIS_URL and _cfg_nube("REDIS_HOST") is None:
        log_evento('INFO', "Streamlit Cloud: sin Redis configurado, la simulación corre sin orquestador de tareas.")
        return None
    if not REDIS_DISPONIBLE:
        log_evento('WARNING', "Librería redis no instalada: la app sigue sin orquestación de tareas.")
        return None
    try:
        if REDIS_URL:
            cliente = redis.Redis.from_url(REDIS_URL, socket_connect_timeout=2, socket_timeout=2,
                                           decode_responses=True)
            destino = "REDIS_URL"
        else:
            cliente = redis.Redis(
                host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, password=REDIS_PASSWORD, ssl=REDIS_SSL,
                socket_connect_timeout=2, socket_timeout=2, decode_responses=True
            )
            destino = f"{REDIS_HOST}:{REDIS_PORT}"
        cliente.ping()
        log_evento('INFO', f"Redis conectado en {destino}: orquestador de tareas activo.")
        return cliente
    except Exception as e:
        log_evento('WARNING', f"Redis no disponible ({e.__class__.__name__}): la app sigue sin orquestación de tareas.")
        return None

def adquirir_lock_simulacion(cliente, propietario, ttl=REDIS_LOCK_TTL):
    """Toma el lock de orquestación para evitar que dos simulaciones pesadas
    corran al mismo tiempo y dejen la app 'pegada'. El TTL libera el lock solo
    si el proceso muere sin avisar."""
    if cliente is None:
        return True  # sin Redis disponible no se bloquea la ejecución
    return bool(cliente.set(REDIS_LOCK_KEY, propietario, nx=True, ex=ttl))

def liberar_lock_simulacion(cliente, propietario):
    """Libera el lock solo si sigue siendo el dueño actual del proceso."""
    if cliente is None:
        return
    if cliente.get(REDIS_LOCK_KEY) == propietario:
        cliente.delete(REDIS_LOCK_KEY)

def actualizar_estado_orquestador(cliente, estado, progreso=0, detalle=""):
    """Publica el estado/progreso del proceso en Redis para poder consultarlo
    sin depender de que la sesión de Streamlit siga viva."""
    log_evento('INFO', f"Simulación [{estado}] {progreso}%: {detalle}")
    _logs_refrescar()
    if cliente is None:
        return
    cliente.set(REDIS_ESTADO_KEY, json.dumps({
        "estado": estado, "progreso": progreso, "detalle": detalle, "timestamp": time.time()
    }))

def registrar_tarea_historial(cliente, id_proceso, evento, detalle=""):
    """Encola en Redis un registro de la tarea (inicio/fin) para trazabilidad
    del orquestador."""
    log_evento('INFO', f"Tarea {id_proceso[:8]} {evento}: {detalle}")
    _logs_refrescar()
    if cliente is None:
        return
    cliente.lpush(REDIS_HISTORIAL_KEY, json.dumps({
        "id_proceso": id_proceso, "evento": evento, "detalle": detalle, "timestamp": time.time()
    }))
    cliente.ltrim(REDIS_HISTORIAL_KEY, 0, 49)  # conserva solo las últimas 50 tareas

# Funciones auxiliares

def r2(valor):
    """Redondeo estándar del reporte: máximo 2 decimales.
    Si el valor absoluto es menor a 0.01 (se vería como 0.00 con 2 decimales),
    usa 2 cifras significativas en su lugar (ej. 0.0033445 -> 0.0033) para no
    perder información en valores muy pequeños. Vectorizado: acepta escalares,
    listas, arrays de numpy o Series/columnas de pandas.
    """
    def _r2_escalar(x):
        if x is None:
            return x
        try:
            if isinstance(x, (float, int, np.floating, np.integer)):
                if pd.isna(x):
                    return x
                x = float(x)
                if x == 0:
                    return 0.0
                if abs(x) < 0.01:
                    exponente = math.floor(math.log10(abs(x)))
                    decimales_sig = -exponente + 1  # 2 cifras significativas
                    return round(x, decimales_sig)
                return round(x, 2)
        except (TypeError, ValueError):
            pass
        return x

    if isinstance(valor, (pd.Series,)):
        return valor.apply(_r2_escalar)
    if isinstance(valor, np.ndarray):
        return np.array([_r2_escalar(v) for v in valor])
    if isinstance(valor, (list, tuple)):
        return [_r2_escalar(v) for v in valor]
    return _r2_escalar(valor)


def fmt2(valor):
    """Formatea un número como string usando la misma regla de r2(), para
    reemplazar los f-strings tipo :.2f/:.3f/:.4f/:.6f del reporte.
    Máximo 2 decimales para valores normales; para valores muy pequeños
    (< 0.01 en valor absoluto) conserva las cifras significativas que ya
    calculó r2() en vez de truncarlas a 0.00 (ej. 0.0033445 -> "0.0033")."""
    v = r2(valor)
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "N/A"
    if not isinstance(v, (float, int, np.floating, np.integer)):
        return str(v)
    v = float(v)
    if v != 0 and abs(v) < 0.01:
        texto = f"{v:.10f}".rstrip('0').rstrip('.')
        return texto if texto else "0"
    return f"{v:.2f}"


def percentiles_reporte(arr):
    """Calcula el set estándar de percentiles usados en el reporte:
    P0 (mínimo), P10, P30, P50, Esperanza (promedio), P70, P90, P100 (máximo)."""
    arr = np.asarray(arr, dtype=float)
    return {
        'p0': r2(np.min(arr)),
        'p10': r2(np.percentile(arr, 10)),
        'p30': r2(np.percentile(arr, 30)),
        'p50': r2(np.percentile(arr, 50)),
        'esperanza': r2(np.mean(arr)),
        'p70': r2(np.percentile(arr, 70)),
        'p90': r2(np.percentile(arr, 90)),
        'p100': r2(np.max(arr)),
    }


def graficar_convergencia_montecarlo(valores, titulo="Convergencia Monte Carlo",
                                      n_checkpoints=60, n_repeticiones=15):
    """
    Grafico de convergencia: muestra como se estabiliza el promedio acumulado del
    AVANCE FINAL de un frente a medida que se van "sumando" simulaciones (1, 2, 3...
    hasta N). Se repite el calculo con distintos barajados (bootstrap del orden) para
    dejar claro que la estabilizacion no depende del orden en que llegaron los datos,
    sino de la cantidad de simulaciones acumuladas.
    """
    valores = np.asarray(valores, dtype=float)
    n = len(valores)
    checkpoints = np.unique(np.linspace(5, n, min(n_checkpoints, n)).astype(int))

    fig = go.Figure()
    for _ in range(n_repeticiones):
        muestra = np.random.permutation(valores)
        medias_acumuladas = [np.mean(muestra[:k]) for k in checkpoints]
        fig.add_trace(go.Scatter(
            x=checkpoints, y=medias_acumuladas, mode='lines',
            line=dict(width=1, color='rgba(31,119,180,0.35)'),
            showlegend=False, hoverinfo='skip'
        ))

    promedio_final = np.mean(valores)
    fig.add_hline(
        y=promedio_final, line_dash="dash", line_color="red",
        annotation_text=f"Promedio con N={n}: {promedio_final:.2f} m"
    )
    fig.update_layout(
        title=titulo,
        xaxis_title="Número de simulaciones acumuladas",
        yaxis_title="Avance Final Promedio Acumulado (m)",
        height=450
    )
    return fig


def graficar_convergencia_percentiles(valores, percentiles=(10, 50, 90),
                                       titulo="Convergencia de Percentiles",
                                       n_checkpoints=60):
    """
    Complementa el grafico de convergencia del promedio: muestra como se estabilizan
    los percentiles P10/P50/P90 (los que realmente se usan en el reporte) a medida
    que aumenta el numero de simulaciones acumuladas.
    """
    valores = np.asarray(valores, dtype=float)
    n = len(valores)
    checkpoints = np.unique(np.linspace(10, n, min(n_checkpoints, n)).astype(int))

    fig = go.Figure()
    colores = {10: 'rgb(255,127,14)', 50: 'rgb(31,119,180)', 90: 'rgb(44,160,44)'}
    for p in percentiles:
        valores_pct = [np.percentile(valores[:k], p) for k in checkpoints]
        fig.add_trace(go.Scatter(
            x=checkpoints, y=valores_pct, mode='lines+markers',
            name=f'P{p}', line=dict(width=2, color=colores.get(p, None))
        ))

    fig.update_layout(
        title=titulo,
        xaxis_title="Número de simulaciones acumuladas",
        yaxis_title="Avance Final (m)",
        height=450,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
    )
    return fig


def calcular_error_relativo_convergencia(valores):
    """
    Calcula el error estandar relativo de la media con el N total disponible.
    Sirve como criterio numerico simple para decidir si el N de simulaciones ya
    es suficiente (regla practica: < 2% se considera bien convergido).
    """
    valores = np.asarray(valores, dtype=float)
    n = len(valores)
    media = np.mean(valores)
    error_estandar = np.std(valores, ddof=1) / np.sqrt(n)
    error_relativo_pct = (error_estandar / media) * 100 if media != 0 else 0.0
    return media, error_estandar, error_relativo_pct


def parse_tiempo(tiempo_str):
    """Parsea el string de tiempo a dict o float"""
    if isinstance(tiempo_str, (int, float)):
        return tiempo_str
    try:
        return ast.literal_eval(tiempo_str)
    except:
        return tiempo_str

def validar_hojas_y_columnas(excel_file, required_schema, nombre_archivo):
    """Valida hojas y columnas minimas antes de ejecutar la simulacion."""
    errors = []
    warnings = []
    xls = pd.ExcelFile(excel_file)
    missing_sheets = [sheet for sheet in required_schema if sheet not in xls.sheet_names]
    if missing_sheets:
        errors.append(f"{nombre_archivo}: faltan hojas {', '.join(missing_sheets)}.")
        return {}, errors, warnings

    data = {}
    for sheet, required_cols in required_schema.items():
        df = xls.parse(sheet)
        data[sheet] = df
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"{nombre_archivo}/{sheet}: faltan columnas {', '.join(missing_cols)}.")

    return data, errors, warnings

def validar_datos_operacion(df_actividades, df_recursos, df_frentes):
    errors = []
    warnings = []

    if df_actividades.empty:
        errors.append("La hoja Actividades esta vacia.")
    if df_recursos.empty:
        errors.append("La hoja Recursos esta vacia.")
    if df_frentes.empty:
        errors.append("La hoja Frentes esta vacia.")

    if 'Distribucion' in df_actividades.columns:
        invalid_dist = sorted(set(df_actividades['Distribucion'].dropna()) - VALID_DISTRIBUTIONS)
        if invalid_dist:
            warnings.append(
                "Distribuciones no reconocidas: "
                + ", ".join(map(str, invalid_dist))
                + ". Se simularan como duracion 1.0 h si no se corrigen."
            )

    for col in ['Xi', 'Yi', 'Xf', 'Yf']:
        if col in df_frentes.columns and df_frentes[col].isna().any():
            errors.append(f"La hoja Frentes tiene coordenadas vacias en la columna {col}.")

    return errors, warnings

def cargar_y_validar_inputs(archivo_main, archivo_restricciones):
    data_main, errors_main, warnings_main = validar_hojas_y_columnas(
        archivo_main, SHEETS_OPERACION, "Archivo 1 Operacion"
    )
    data_rest, errors_rest, warnings_rest = validar_hojas_y_columnas(
        archivo_restricciones, SHEETS_RESTRICCIONES, "Archivo 2 Restricciones"
    )

    errors = errors_main + errors_rest
    warnings = warnings_main + warnings_rest
    if errors:
        return None, errors, warnings

    df_actividades = data_main['Actividades'].copy()
    df_recursos = data_main['Recursos'].copy()
    df_frentes = data_main['Frentes'].copy()
    df_actividades['Tiempo'] = df_actividades['Tiempo'].apply(parse_tiempo)

    data_errors, data_warnings = validar_datos_operacion(df_actividades, df_recursos, df_frentes)
    errors.extend(data_errors)
    warnings.extend(data_warnings)

    return {
        'df_actividades': df_actividades,
        'df_recursos': df_recursos,
        'df_frentes': df_frentes,
        'df_fh': data_rest['FrenteHundimiento'].copy(),
        'df_oc': data_rest['ObrasCiviles'].copy(),
        'df_res': data_rest['Restriccion'].copy()
    }, errors, warnings

def restriction_radius(tipo, default_radius, radios_por_tipo=None):
    if radios_por_tipo is None:
        radios_por_tipo = DEFAULT_RESTRICTION_RADII
    if pd.isna(tipo):
        return float(default_radius)
    return float(radios_por_tipo.get(str(tipo), default_radius))

def columna_seccion(df_frentes):
    for candidate in ['Sección', 'Seccion', 'SecciÃ³n']:
        if candidate in df_frentes.columns:
            return candidate
    return None

# ---- Filtrado por Nivel (cacheado) ----
# frentes_nivel y actividades_nivel son las tablas base que casi todo el resto de la app
# lee (Menús 5 a 16). Sin cache, Streamlit repetía este filtrado del DataFrame completo
# en CADA rerun -aunque el Nivel seleccionado no cambiara- por cualquier interacción en
# cualquier parte de la app. Los DataFrames grandes se reciben con guion bajo
# (_df_actividades, _df_recursos, _df_frentes) para que Streamlit no intente hashearlos
# completos en cada llamada; la validez del cache depende solo de 'nivel_seleccionado' y
# de 'data_version' (que únicamente cambia cuando se cargan archivos nuevos en el Menú 1).
@st.cache_data(show_spinner=False)
def filtrar_actividades_nivel(_df_actividades, nivel_seleccionado, data_version):
    """Filtra la hoja Actividades por Nivel."""
    return _df_actividades[_df_actividades['Nivel'] == nivel_seleccionado].copy()


@st.cache_data(show_spinner=False)
def filtrar_recursos_nivel(_df_recursos, nivel_seleccionado, data_version):
    """Filtra la hoja Recursos por Nivel."""
    return _df_recursos[_df_recursos['Nivel'] == nivel_seleccionado].copy()


@st.cache_data(show_spinner=False)
def filtrar_frentes_nivel(_df_frentes, nivel_seleccionado, data_version):
    """Filtra la hoja Frentes por Nivel."""
    return _df_frentes[_df_frentes['Nivel'] == nivel_seleccionado].copy()


def distancia_minima_a_puntos(df_puntos, x, y):
    if df_puntos is None or df_puntos.empty or not {'X', 'Y'}.issubset(df_puntos.columns):
        return float('inf')
    return min(calcular_distancia(x, y, row['X'], row['Y']) for _, row in df_puntos.iterrows())

def ordenar_frentes_por_prioridad(frentes_df, frentes_seleccionados, df_fh, df_oc, ruta_critica):
    """Orden de prioridad de los frentes a simular.

    Si hay ruta crítica (opcional), esos frentes van PRIMERO y en EXACTAMENTE el orden que definió
    el usuario (1º, 2º, 3º...). Los frentes restantes (no incluidos en la ruta) se ordenan después
    por cercanía a FH y a obras civiles."""
    ruta_ordenada = []
    for f in (ruta_critica or []):
        if f in frentes_seleccionados and f not in ruta_ordenada:
            ruta_ordenada.append(f)
    resto = []
    for frente in frentes_seleccionados:
        if frente in ruta_ordenada:
            continue
        row = frentes_df[frentes_df['Frentes'] == frente].iloc[0]
        fh_distance = distancia_minima_a_puntos(df_fh, row['Xi'], row['Yi'])
        oc_distance = distancia_minima_a_puntos(df_oc, row['Xi'], row['Yi'])
        resto.append((fh_distance, oc_distance, frente))
    return ruta_ordenada + [item[-1] for item in sorted(resto)]

def avance_efectivo_frente(frente, metros_avance_default, frentes_info, mapa_restriccion_geologica=None):
    metros = frentes_info.get(frente, {}).get('metros_por_ciclo', metros_avance_default)
    if mapa_restriccion_geologica and frente in mapa_restriccion_geologica:
        castigo_pct = mapa_restriccion_geologica[frente].get('castigo_avance', 0)
        metros *= 1 + (castigo_pct / 100.0)
    return max(0.0, float(metros))

DIST_OBJ_SCIPY = {
    'fisk': stats.fisk, 'normal': stats.norm, 'weibull': stats.weibull_min,
    'gamma': stats.gamma, 'lognormal': stats.lognorm, 'kstwobign': stats.kstwobign,
    'rayleigh': stats.rayleigh, 'foldcauchy': stats.foldcauchy, 'foldnorm': stats.foldnorm,
    'ncx2': stats.ncx2, 'burr': stats.burr, 'loglaplace': stats.loglaplace,
    'maxwell': stats.maxwell, 'nakagami': stats.nakagami,
}


def _kwargs_scipy_desde_params(dist_params, distribucion):
    """Traduce el dict de parametros (nombres de la UI) a los kwargs que espera
    cada funcion de scipy.stats, sin volver a escribir el mapeo en cada lugar."""
    if distribucion == 'normal':
        return {'loc': dist_params['mean'], 'scale': dist_params['std']}
    elif distribucion == 'fisk':
        return {'c': dist_params['c'], 'loc': dist_params['loc'], 'scale': dist_params['scale']}
    elif distribucion == 'weibull':
        return {'c': dist_params['c'], 'loc': dist_params['loc'], 'scale': dist_params['scale']}
    elif distribucion == 'gamma':
        return {'a': dist_params['a'], 'loc': dist_params['loc'], 'scale': dist_params['scale']}
    elif distribucion == 'lognormal':
        return {'s': dist_params['s'], 'loc': dist_params['loc'], 'scale': dist_params['scale']}
    elif distribucion == 'kstwobign':
        return {'loc': dist_params['loc'], 'scale': dist_params['scale']}
    elif distribucion == 'rayleigh':
        return {'loc': dist_params.get('loc', 0), 'scale': dist_params.get('scale', 1)}
    elif distribucion == 'foldcauchy':
        return {'c': dist_params.get('c', 0), 'loc': dist_params.get('loc', 0), 'scale': dist_params.get('scale', 1)}
    elif distribucion == 'foldnorm':
        return {'c': dist_params.get('c', 0), 'loc': dist_params.get('loc', 0), 'scale': dist_params.get('scale', 1)}
    elif distribucion == 'ncx2':
        return {'df': dist_params.get('df', 2), 'nc': dist_params.get('nc', 0), 'loc': dist_params.get('loc', 0), 'scale': dist_params.get('scale', 1)}
    elif distribucion == 'burr':
        return {'c': dist_params['c'], 'd': dist_params['d'], 'loc': dist_params.get('loc', 0), 'scale': dist_params.get('scale', 1)}
    elif distribucion == 'loglaplace':
        return {'c': dist_params['c'], 'loc': dist_params.get('loc', 0), 'scale': dist_params.get('scale', 1)}
    elif distribucion == 'maxwell':
        return {'loc': dist_params.get('loc', 0), 'scale': dist_params.get('scale', 1)}
    elif distribucion == 'nakagami':
        return {'nu': dist_params['nu'], 'loc': dist_params.get('loc', 0), 'scale': dist_params.get('scale', 1)}
    return {}


def generar_tiempos_batch(dist_params, distribucion, n_samples):
    """Genera múltiples tiempos aleatorios de una vez (duraciones de actividad, en horas).

    Envoltorio delgado sobre _generar_tiempos_batch_raw que aplica un clamp a 0 como
    red de seguridad general: algunas distribuciones con loc negativo (ej. loglaplace
    con loc=-25.03 en PF/Jumbo_F de NT) tienen una probabilidad pequeña pero no nula
    de generar una duracion negativa. Sin este clamp, una duracion negativa liberaba
    el recurso "antes" de empezar la tarea y el evento ni siquiera quedaba registrado
    en la traza. Se aplica sobre TODAS las distribuciones, no solo sobre el caso
    puntual detectado, para blindar cualquier otra combinacion de parametros futura
    que pueda producir el mismo efecto."""
    resultado = _generar_tiempos_batch_raw(dist_params, distribucion, n_samples)
    return np.maximum(0.0, resultado)


def _generar_tiempos_batch_raw(dist_params, distribucion, n_samples):
    """Genera múltiples tiempos aleatorios de una vez.

    Si dist_params trae las claves '_trunc_lo'/'_trunc_hi' (filtro de rango aplicado
    como TRUNCAMIENTO real, no solo como indicador visual), los valores generados
    quedan garantizados dentro de [_trunc_lo, _trunc_hi]: se invierte la CDF de la
    distribucion base restringida a ese tramo (muestreo uniforme de u en
    [cdf(lo), cdf(hi)] y ppf(u)), que es exacto y no depende de rechazo/reintentos."""
    trunc_lo = dist_params.get('_trunc_lo') if isinstance(dist_params, dict) else None
    trunc_hi = dist_params.get('_trunc_hi') if isinstance(dist_params, dict) else None

    if distribucion == 'Cte' or isinstance(dist_params, (int, float)):
        valor = float(dist_params) if isinstance(dist_params, (int, float)) else 1.0
        return np.full(n_samples, valor)

    try:
        if trunc_lo is not None and trunc_hi is not None and distribucion in DIST_OBJ_SCIPY:
            dist_obj = DIST_OBJ_SCIPY[distribucion]
            kwargs = _kwargs_scipy_desde_params(dist_params, distribucion)
            cdf_lo = dist_obj.cdf(trunc_lo, **kwargs)
            cdf_hi = dist_obj.cdf(trunc_hi, **kwargs)
            if np.isfinite(cdf_lo) and np.isfinite(cdf_hi) and cdf_hi > cdf_lo:
                u = np.random.uniform(cdf_lo, cdf_hi, size=n_samples)
                vals = dist_obj.ppf(u, **kwargs)
            else:
                # Rango en una cola sin masa numerica (o cdf invalida): no hay forma de
                # invertir la CDF; se muestrea uniforme DENTRO del rango en vez de fallar.
                vals = np.random.uniform(trunc_lo, trunc_hi, size=n_samples)
            # Garantia dura (inclusiva): ppf puede devolver +-inf/NaN en los extremos
            # o salirse por redondeo; nunca se entrega nada fuera de [lo, hi].
            vals = np.where(np.isfinite(vals), vals, np.random.uniform(trunc_lo, trunc_hi, size=n_samples))
            return np.clip(vals, trunc_lo, trunc_hi)

        if distribucion == 'fisk':
            return stats.fisk.rvs(c=dist_params['c'], loc=dist_params['loc'], 
                                 scale=dist_params['scale'], size=n_samples)
        elif distribucion == 'normal':
            return stats.norm.rvs(loc=dist_params['mean'], scale=dist_params['std'], size=n_samples)
        elif distribucion == 'weibull':
            return stats.weibull_min.rvs(c=dist_params['c'], loc=dist_params['loc'], 
                                         scale=dist_params['scale'], size=n_samples)
        elif distribucion == 'gamma':
            return stats.gamma.rvs(a=dist_params['a'], loc=dist_params['loc'], 
                                  scale=dist_params['scale'], size=n_samples)
        elif distribucion == 'lognormal':
            return stats.lognorm.rvs(s=dist_params['s'], loc=dist_params['loc'], 
                                    scale=dist_params['scale'], size=n_samples)
        elif distribucion == 'kstwobign':
            return stats.kstwobign.rvs(loc=dist_params['loc'], scale=dist_params['scale'], size=n_samples)
        elif distribucion == 'rayleigh':
            return stats.rayleigh.rvs(loc=dist_params.get('loc', 0), scale=dist_params.get('scale', 1), size=n_samples)
        elif distribucion == 'foldcauchy':
            return stats.foldcauchy.rvs(c=dist_params.get('c', 0), loc=dist_params.get('loc', 0), scale=dist_params.get('scale', 1), size=n_samples)
        elif distribucion == 'foldnorm':
            return stats.foldnorm.rvs(c=dist_params.get('c', 0), loc=dist_params.get('loc', 0), scale=dist_params.get('scale', 1), size=n_samples)
        elif distribucion == 'ncx2':
            return stats.ncx2.rvs(df=dist_params.get('df', 2), nc=dist_params.get('nc', 0), loc=dist_params.get('loc', 0), scale=dist_params.get('scale', 1), size=n_samples)
        elif distribucion == 'burr':
            return stats.burr.rvs(c=dist_params['c'], d=dist_params['d'], loc=dist_params.get('loc', 0), scale=dist_params.get('scale', 1), size=n_samples)
        elif distribucion == 'loglaplace':
            return stats.loglaplace.rvs(c=dist_params['c'], loc=dist_params.get('loc', 0), scale=dist_params.get('scale', 1), size=n_samples)
        elif distribucion == 'maxwell':
            return stats.maxwell.rvs(loc=dist_params.get('loc', 0), scale=dist_params.get('scale', 1), size=n_samples)
        elif distribucion == 'nakagami':
            return stats.nakagami.rvs(nu=dist_params['nu'], loc=dist_params.get('loc', 0), scale=dist_params.get('scale', 1), size=n_samples)
        else:
            return np.ones(n_samples)
    except Exception as e:
        return np.ones(n_samples)

def calcular_pdf_teorica(dist_params, distribucion, x_vals):
    """Calcula la densidad de probabilidad (PDF) analitica exacta de la distribucion
    de entrada evaluada sobre x_vals. El area bajo esta curva integra a 1 por
    definicion (a diferencia de un histograma, que depende del binning).

    Si dist_params trae '_trunc_lo'/'_trunc_hi', devuelve la PDF TRUNCADA y
    RENORMALIZADA (0 fuera del rango, y dentro del rango la pdf original dividida
    por el area que queda entre esos limites, para que vuelva a integrar a 1)."""
    trunc_lo = dist_params.get('_trunc_lo') if isinstance(dist_params, dict) else None
    trunc_hi = dist_params.get('_trunc_hi') if isinstance(dist_params, dict) else None
    try:
        if distribucion == 'Cte' or isinstance(dist_params, (int, float)):
            return np.zeros_like(x_vals)
        elif distribucion == 'fisk':
            pdf_vals = stats.fisk.pdf(x_vals, c=dist_params['c'], loc=dist_params['loc'],
                                  scale=dist_params['scale'])
        elif distribucion == 'normal':
            pdf_vals = stats.norm.pdf(x_vals, loc=dist_params['mean'], scale=dist_params['std'])
        elif distribucion == 'weibull':
            pdf_vals = stats.weibull_min.pdf(x_vals, c=dist_params['c'], loc=dist_params['loc'],
                                         scale=dist_params['scale'])
        elif distribucion == 'gamma':
            pdf_vals = stats.gamma.pdf(x_vals, a=dist_params['a'], loc=dist_params['loc'],
                                   scale=dist_params['scale'])
        elif distribucion == 'lognormal':
            pdf_vals = stats.lognorm.pdf(x_vals, s=dist_params['s'], loc=dist_params['loc'],
                                     scale=dist_params['scale'])
        elif distribucion == 'kstwobign':
            pdf_vals = stats.kstwobign.pdf(x_vals, loc=dist_params['loc'], scale=dist_params['scale'])
        elif distribucion == 'rayleigh':
            pdf_vals = stats.rayleigh.pdf(x_vals, loc=dist_params.get('loc', 0), scale=dist_params.get('scale', 1))
        elif distribucion == 'foldcauchy':
            pdf_vals = stats.foldcauchy.pdf(x_vals, c=dist_params.get('c', 0), loc=dist_params.get('loc', 0),
                                        scale=dist_params.get('scale', 1))
        elif distribucion == 'foldnorm':
            pdf_vals = stats.foldnorm.pdf(x_vals, c=dist_params.get('c', 0), loc=dist_params.get('loc', 0),
                                      scale=dist_params.get('scale', 1))
        elif distribucion == 'ncx2':
            pdf_vals = stats.ncx2.pdf(x_vals, df=dist_params.get('df', 2), nc=dist_params.get('nc', 0),
                                  loc=dist_params.get('loc', 0), scale=dist_params.get('scale', 1))
        elif distribucion == 'burr':
            pdf_vals = stats.burr.pdf(x_vals, c=dist_params['c'], d=dist_params['d'],
                                  loc=dist_params.get('loc', 0), scale=dist_params.get('scale', 1))
        elif distribucion == 'loglaplace':
            pdf_vals = stats.loglaplace.pdf(x_vals, c=dist_params['c'], loc=dist_params.get('loc', 0),
                                        scale=dist_params.get('scale', 1))
        elif distribucion == 'maxwell':
            pdf_vals = stats.maxwell.pdf(x_vals, loc=dist_params.get('loc', 0), scale=dist_params.get('scale', 1))
        elif distribucion == 'nakagami':
            pdf_vals = stats.nakagami.pdf(x_vals, nu=dist_params['nu'], loc=dist_params.get('loc', 0),
                                      scale=dist_params.get('scale', 1))
        else:
            return np.zeros_like(x_vals)

        if trunc_lo is not None and trunc_hi is not None and distribucion in DIST_OBJ_SCIPY:
            dist_obj = DIST_OBJ_SCIPY[distribucion]
            kwargs = _kwargs_scipy_desde_params(dist_params, distribucion)
            area_tramo = dist_obj.cdf(trunc_hi, **kwargs) - dist_obj.cdf(trunc_lo, **kwargs)
            fuera_rango = (x_vals < trunc_lo) | (x_vals > trunc_hi)
            pdf_vals = np.where(fuera_rango, 0.0, pdf_vals / area_tramo if area_tramo > 0 else 0.0)

        return pdf_vals
    except Exception:
        return np.zeros_like(x_vals)

def generar_velocidad_batch(dist_params, distribucion, n_samples, random_state=None):
    """Genera múltiples velocidades aleatorias de una vez.

    Si dist_params trae '_trunc_lo'/'_trunc_hi' (filtro de rango elegido en la
    vista previa del Menú 6: "Mayor a"/"Menor a"/"Entre"), las velocidades
    generadas quedan garantizadas dentro de ese rango, con la misma técnica
    exacta (inversión de la CDF restringida) que ya usa _generar_tiempos_batch_raw
    para los tiempos de actividad.

    random_state: opcional, instancia de numpy.random.Generator (ej.
    np.random.default_rng(seed)). Se la pasa tal cual a los .rvs() de scipy.
    Se usa para paralelizar el precálculo de velocidades por recurso con hilos
    (ver simular_avance_con_transporte): cada hilo recibe su PROPIO Generator
    independiente en vez de compartir el generador global de NumPy, que no es
    thread-safe (varios hilos escribiendo/leyendo su estado interno a la vez
    pueden producir muestras correlacionadas o no reproducibles). Si se omite
    (None), el comportamiento es exactamente el de antes: usa el generador
    global de NumPy."""
    if distribucion == 'Constante' or isinstance(dist_params, (int, float)):
        valor = float(dist_params) if isinstance(dist_params, (int, float)) else 15.0
        return np.full(n_samples, valor)

    trunc_lo = dist_params.get('_trunc_lo') if isinstance(dist_params, dict) else None
    trunc_hi = dist_params.get('_trunc_hi') if isinstance(dist_params, dict) else None
    _uniform = random_state.uniform if random_state is not None else np.random.uniform

    try:
        if trunc_lo is not None and trunc_hi is not None and distribucion in DIST_OBJ_SCIPY:
            dist_obj = DIST_OBJ_SCIPY[distribucion]
            kwargs = _kwargs_scipy_desde_params(dist_params, distribucion)
            cdf_lo = dist_obj.cdf(trunc_lo, **kwargs)
            cdf_hi = dist_obj.cdf(trunc_hi, **kwargs)
            u = _uniform(cdf_lo, cdf_hi, size=n_samples)
            return np.abs(dist_obj.ppf(u, **kwargs))

        if distribucion == 'normal':
            return np.abs(stats.norm.rvs(loc=dist_params['mean'], scale=dist_params['std'], size=n_samples, random_state=random_state))
        elif distribucion == 'weibull':
            return np.abs(stats.weibull_min.rvs(c=dist_params['c'], loc=dist_params.get('loc', 0),
                                                 scale=dist_params['scale'], size=n_samples, random_state=random_state))
        elif distribucion == 'gamma':
            return np.abs(stats.gamma.rvs(a=dist_params['a'], loc=dist_params.get('loc', 0),
                                          scale=dist_params['scale'], size=n_samples, random_state=random_state))
        elif distribucion == 'lognormal':
            return np.abs(stats.lognorm.rvs(s=dist_params['s'], loc=dist_params.get('loc', 0),
                                            scale=dist_params['scale'], size=n_samples, random_state=random_state))
        elif distribucion == 'fisk':
            return np.abs(stats.fisk.rvs(c=dist_params['c'], loc=dist_params.get('loc', 0),
                                         scale=dist_params['scale'], size=n_samples, random_state=random_state))
        elif distribucion == 'kstwobign':
            return np.abs(stats.kstwobign.rvs(loc=dist_params.get('loc', 0),
                                              scale=dist_params['scale'], size=n_samples, random_state=random_state))
        elif distribucion == 'rayleigh':
            return stats.rayleigh.rvs(loc=dist_params.get('loc', 0),
                                      scale=dist_params['scale'], size=n_samples, random_state=random_state)
        elif distribucion == 'foldcauchy':
            return stats.foldcauchy.rvs(c=dist_params.get('c', 0), loc=dist_params.get('loc', 0),
                                        scale=dist_params['scale'], size=n_samples, random_state=random_state)
        elif distribucion == 'foldnorm':
            return stats.foldnorm.rvs(c=dist_params.get('c', 0), loc=dist_params.get('loc', 0),
                                      scale=dist_params['scale'], size=n_samples, random_state=random_state)
        elif distribucion == 'ncx2':
            return np.abs(stats.ncx2.rvs(df=dist_params.get('df', 2), nc=dist_params.get('nc', 0),
                                         loc=dist_params.get('loc', 0), scale=dist_params['scale'],
                                         size=n_samples, random_state=random_state))
        elif distribucion == 'burr':
            return np.abs(stats.burr.rvs(c=dist_params['c'], d=dist_params['d'], loc=dist_params.get('loc', 0),
                                         scale=dist_params['scale'], size=n_samples, random_state=random_state))
        elif distribucion == 'loglaplace':
            return np.abs(stats.loglaplace.rvs(c=dist_params['c'], loc=dist_params.get('loc', 0),
                                               scale=dist_params['scale'], size=n_samples, random_state=random_state))
        elif distribucion == 'maxwell':
            return stats.maxwell.rvs(loc=dist_params.get('loc', 0), scale=dist_params['scale'], size=n_samples, random_state=random_state)
        elif distribucion == 'nakagami':
            return np.abs(stats.nakagami.rvs(nu=dist_params['nu'], loc=dist_params.get('loc', 0),
                                             scale=dist_params['scale'], size=n_samples, random_state=random_state))
        else:
            return np.full(n_samples, 15.0)
    except Exception:
        return np.full(n_samples, 15.0)

def vista_previa_pdf_con_filtro(distribucion, params, key_sufijo, unidad="h", titulo_eje_x=None):
    """Grafica la PDF de una distribución (scipy.stats) y agrega un filtro
    interactivo de área bajo la curva ("Mayor a", "Menor a", "Entre"), igual
    al que ya existe en el Menú 8 (Modificar Distribuciones de Actividades).

    Devuelve (x_area_lo, x_area_hi) si el usuario dejó un filtro activo
    ("Ninguna" no cuenta), o None si no hay filtro. El llamador decide qué
    hacer con ese rango (ej. guardarlo como '_trunc_lo'/'_trunc_hi' dentro de
    los propios parámetros, para que la simulación quede truncada a ese
    tramo real, no solo mostrado visualmente).

    'key_sufijo' debe ser único por cada distribución mostrada en la página
    (ej. el nombre del recurso), para que los widgets no choquen entre sí."""
    if distribucion not in DIST_OBJ_SCIPY:
        st.info("No hay curva de densidad para esta distribución.")
        return None

    titulo_eje_x = titulo_eje_x or f"Valor ({unidad})"
    try:
        dist_obj = DIST_OBJ_SCIPY[distribucion]
        dist_kwargs = _kwargs_scipy_desde_params(params, distribucion)

        # Rango de x centrado en la forma de la distribucion, usando los
        # percentiles 0.1% y 99.9% (ppf) en vez de adivinar limites a mano.
        x_lo = dist_obj.ppf(0.001, **dist_kwargs)
        x_hi = dist_obj.ppf(0.999, **dist_kwargs)
        if not np.isfinite(x_lo):
            x_lo = 0.0
        if not np.isfinite(x_hi):
            x_hi = x_lo + 10.0
        if x_hi <= x_lo:
            x_hi = x_lo + 1.0
        x_vals = np.linspace(x_lo, x_hi, 400)
        y_vals = calcular_pdf_teorica(params, distribucion, x_vals)

        fig_preview = go.Figure()
        fig_preview.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode='lines', fill='tozeroy',
            line=dict(color='#1f77b4', width=2),
            name=distribucion
        ))

        col_area1, col_area2 = st.columns([1, 2])
        with col_area1:
            modo_area = st.selectbox(
                "Área bajo la curva a resaltar",
                ["Ninguna", "Mayor a (>)", "Menor a (<)", "Entre (rango)"],
                key=f"modo_area_{key_sufijo}"
            )
        with col_area2:
            x_area_lo, x_area_hi = None, None
            if modo_area == "Mayor a (>)":
                x_area_lo = st.slider(f"Valor de corte ({unidad})", float(x_lo), float(x_hi),
                                       value=float((x_lo + x_hi) / 2), key=f"corte_mayor_{key_sufijo}")
                x_area_hi = x_hi
            elif modo_area == "Menor a (<)":
                x_area_hi = st.slider(f"Valor de corte ({unidad})", float(x_lo), float(x_hi),
                                       value=float((x_lo + x_hi) / 2), key=f"corte_menor_{key_sufijo}")
                x_area_lo = x_lo
            elif modo_area == "Entre (rango)":
                x_area_lo, x_area_hi = st.slider(
                    f"Rango ({unidad})", float(x_lo), float(x_hi),
                    value=(float(x_lo), float(x_hi)), key=f"corte_rango_{key_sufijo}"
                )

        resultado_rango = None
        if modo_area != "Ninguna" and x_area_lo is not None:
            mascara_area = (x_vals >= x_area_lo) & (x_vals <= x_area_hi)
            fig_preview.add_trace(go.Scatter(
                x=x_vals[mascara_area], y=y_vals[mascara_area],
                mode='lines', fill='tozeroy',
                line=dict(width=0), fillcolor='rgba(255,99,71,0.55)',
                name='Área seleccionada'
            ))
            prob_area = dist_obj.cdf(x_area_hi, **dist_kwargs) - dist_obj.cdf(x_area_lo, **dist_kwargs)
            st.caption(
                f"Probabilidad de que el valor caiga en [{x_area_lo:.3f}, {x_area_hi:.3f}] {unidad}: "
                f"**{prob_area*100:.2f}%** del área total."
            )
            resultado_rango = (float(x_area_lo), float(x_area_hi))

        fig_preview.update_layout(
            height=280,
            margin=dict(l=40, r=20, t=20, b=40),
            xaxis_title=titulo_eje_x,
            yaxis_title="Densidad de probabilidad",
            showlegend=False
        )
        st.plotly_chart(fig_preview, width='stretch', key=f"preview_pdf_{key_sufijo}")
        return resultado_rango
    except Exception as e:
        st.warning(f"No fue posible graficar la vista previa con los parámetros actuales: {e}")
        return None


def calcular_distancia(xi, yi, xf, yf):
    """Calcula distancia euclidiana"""
    return math.sqrt((xf - xi)**2 + (yf - yi)**2)

def calcular_posicion_avanzada(xi, yi, xf, yf, metros_avanzados, distancia_total):
    """Calcula la nueva posición después de avanzar ciertos metros en el túnel"""
    if distancia_total == 0:
        return xi, yi
    
    proporcion = metros_avanzados / distancia_total
    nuevo_x = xi + (xf - xi) * proporcion
    nuevo_y = yi + (yf - yi) * proporcion
    
    return nuevo_x, nuevo_y

def check_tunnel_restriction_intersection(Px, Py, Xi, Yi, Xf, Yf, radius):
    """
    Verifica si el segmento de línea (túnel) [Xi, Yi] a [Xf, Yf]
    interseca la circunferencia centrada en [Px, Py] con radio 'radius'.
    """
    
    dist_centro_a_Xi = calcular_distancia(Px, Py, Xi, Yi)
    dist_centro_a_Xf = calcular_distancia(Px, Py, Xf, Yf)
    if dist_centro_a_Xi <= radius or dist_centro_a_Xf <= radius:
        return True
    
    dx = Xf - Xi
    dy = Yf - Yi
    t_den = dx*dx + dy*dy

    if t_den == 0:
        return dist_centro_a_Xi <= radius
    
    t_num = (Px - Xi) * dx + (Py - Yi) * dy
    t = t_num / t_den
    
    if t < 0.0:
        closest_x, closest_y = Xi, Yi
    elif t > 1.0:
        closest_x, closest_y = Xf, Yf
    else:
        closest_x = Xi + t * dx
        closest_y = Yi + t * dy

    distance_to_segment = calcular_distancia(Px, Py, closest_x, closest_y)
    
    return distance_to_segment <= radius

def calcular_demoras_restricciones(df_frentes, df_fh, df_oc, df_res, radio_restriccion, demora_horas, radios_por_tipo=None):
    """
    Calcula las demoras totales (FH + OC) que se aplican a cada túnel por ciclo,
    y las demoras específicas por turno (R).
    """
    
    dict_demoras_constantes = {frente: 0 for frente in df_frentes['Frentes'].unique()}
    list_demoras_por_turno = []
    
    for frente_nombre in df_frentes['Frentes'].unique():
        frente_data = df_frentes[df_frentes['Frentes'] == frente_nombre].iloc[0]
        Xi, Yi, Xf, Yf = frente_data['Xi'], frente_data['Yi'], frente_data['Xf'], frente_data['Yf']
        
        demora_acumulada_constante = 0
        
        # 1. Check FH (Frente Hundimiento)
        impacto_fh = False
        if 'X' in df_fh.columns and 'Y' in df_fh.columns:
            for _, row in df_fh.iterrows():
                Px, Py = row['X'], row['Y']
                if check_tunnel_restriction_intersection(Px, Py, Xi, Yi, Xf, Yf, radio_restriccion):
                    impacto_fh = True
                    break
        if impacto_fh:
            demora_acumulada_constante += demora_horas
            
        # 2. Check OC (Obras Civiles)
        impacto_oc = False
        if 'X' in df_oc.columns and 'Y' in df_oc.columns:
            for _, row in df_oc.iterrows():
                Px, Py = row['X'], row['Y']
                if check_tunnel_restriction_intersection(Px, Py, Xi, Yi, Xf, Yf, radio_restriccion):
                    impacto_oc = True
                    break
        if impacto_oc:
            demora_acumulada_constante += demora_horas

        dict_demoras_constantes[frente_nombre] = demora_acumulada_constante

    # 3. Procesar Restricción (R) - Demora Dependiente del Turno
    if 'X' in df_res.columns and 'Y' in df_res.columns and 'Turno' in df_res.columns and 'Tipo' in df_res.columns:
        tiene_index_i = 'Index_I' in df_res.columns
        tiene_index_frente = 'Index' in df_frentes.columns
        for frente_nombre in df_frentes['Frentes'].unique():
            frente_data = df_frentes[df_frentes['Frentes'] == frente_nombre].iloc[0]
            Xi, Yi, Xf, Yf = frente_data['Xi'], frente_data['Yi'], frente_data['Xf'], frente_data['Yf']
            index_frente = frente_data['Index'] if tiene_index_frente else None

            for _, row in df_res.iterrows():
                Px, Py = row['X'], row['Y']
                turno = int(row['Turno']) 
                tipo = f"Demora por Restricción ({row['Tipo']})"
                radio_turno = restriction_radius(row.get('Tipo'), radio_restriccion, radios_por_tipo)

                # Si la fila de Restriccion trae Index_I (referencia directa al 'Index' del
                # frente en la hoja Frentes), la restriccion se aplica SOLO a ese frente
                # puntual, sin depender del radio de impacto geometrico X/Y. Si Index_I
                # viene vacio (como en el input original), se mantiene el chequeo
                # geometrico de siempre — comportamiento 100% retrocompatible.
                index_i_valor = row.get('Index_I') if tiene_index_i else None
                if tiene_index_i and pd.notna(index_i_valor):
                    try:
                        aplica = (
                            index_frente is not None and pd.notna(index_frente)
                            and int(float(index_i_valor)) == int(float(index_frente))
                        )
                    except (ValueError, TypeError):
                        aplica = False
                else:
                    aplica = check_tunnel_restriction_intersection(Px, Py, Xi, Yi, Xf, Yf, radio_turno)

                if aplica:
                    list_demoras_por_turno.append({
                        'Frente': frente_nombre,
                        'Turno': turno,
                        'Demora': demora_horas,
                        'Tipo': tipo,
                        'Radio': radio_turno,
                        'X': Px,
                        'Y': Py
                    })
    
    return dict_demoras_constantes, list_demoras_por_turno


# ----------------------------------------------------------------------------------
# Bloqueo TOTAL de túneles (restricción dura, tipo "shift_State = False" de David)
# ----------------------------------------------------------------------------------
# calcular_demoras_restricciones (arriba) NO detiene el turno: solo demora el inicio
# del ciclo unas horas fijas y el frente sigue avanzando despues. Eso es correcto como
# interpretacion de "una tronadura cercana demora, no necesariamente detiene el turno
# completo", pero es una decision de negocio distinta a la del motor original de David,
# donde un frente dentro del radio de una restriccion en un turno dado queda con
# avance = 0 ESE turno (bloqueo total), sin trabajar nada.
#
# Las funciones de esta seccion permiten, ADEMAS de la demora en horas de arriba,
# marcar explicitamente ciertos tuneles como bloqueados por completo durante una
# ventana de tiempo (o toda la planificacion), replicando ese comportamiento cuando
# el usuario lo necesite, sin tener que elegir una unica interpretacion global.

def frente_en_bloqueo_total(frente, tiempo, bloqueos_totales_frente):
    """
    Indica si 'frente' esta, en este instante, dentro de una ventana de BLOQUEO TOTAL
    (avance 0, sin ejecutar ninguna actividad ni consumir ningun recurso) configurada
    en el Menu 14 ('Bloqueo Total de Túneles').

    A diferencia de la demora en horas (que solo retrasa el inicio del ciclo y el
    frente sigue avanzando), un bloqueo total congela el frente por completo durante
    toda la ventana: no arranca ninguna secuencia/actividad, y los recursos que le
    habrian correspondido quedan libres para otros frentes durante ese lapso.

    Retorna el instante (hora de simulacion) en que termina el bloqueo si 'tiempo' cae
    dentro de una ventana; retorna None si el frente esta libre en ese instante.
    """
    for inicio, fin in bloqueos_totales_frente.get(frente, []):
        if inicio <= tiempo < fin:
            return fin
    return None


def _fusionar_intervalos(intervalos):
    """Ordena y fusiona una lista de tuplas (inicio, fin) que se solapen o sean
    contiguas, para que frente_en_bloqueo_total no itere de mas ni se dupliquen
    ventanas superpuestas configuradas por error (ej. dos filas para el mismo
    tunel con rangos que se cruzan)."""
    if not intervalos:
        return []
    ordenados = sorted(intervalos, key=lambda iv: iv[0])
    fusionados = [ordenados[0]]
    for inicio, fin in ordenados[1:]:
        ult_inicio, ult_fin = fusionados[-1]
        if inicio <= ult_fin:  # se solapan o son contiguas -> fusionar en una sola ventana
            fusionados[-1] = (ult_inicio, max(ult_fin, fin))
        else:
            fusionados.append((inicio, fin))
    return fusionados


def calcular_bloqueos_tuneles(bloqueos_config, duracion_turno, tiempo_limite):
    """
    Convierte la configuracion de 'Bloqueo Total de Túneles' (una fila por tunel
    bloqueado, definida en la UI del Menu 14) en un diccionario
    {frente: [(inicio_h, fin_h), ...]} listo para pasarle a
    simular_avance_con_transporte.

    Cada elemento de bloqueos_config es un dict con:
      - 'Frente': nombre del tunel/frente a bloquear.
      - 'Modo': uno de
          'Toda la planificación'                  -> bloquea [0, tiempo_limite]
                                                        completo (avance final = 0
                                                        para ese frente).
          'Desde el inicio (N horas/turnos)'        -> bloquea [0, duración] desde
                                                        el instante 0 (IGNORA 'Inicio').
          'En un momento específico'                -> bloquea [inicio, inicio+duración].
          'Entre Inicio y Fin (turno u hora)'       -> bloquea [inicio, fin] usando 'Inicio'
                                                        y 'Fin' directamente (sin pasar por
                                                        una duración calculada); pensado para
                                                        cuando el usuario quiere decir
                                                        literalmente "del turno 18 al 22"
                                                        en vez de "desde el 18, por 5 turnos".
      - 'Unidad': 'Horas' o 'Turnos' (aplica a 'Inicio', 'Fin' y 'Duracion').
      - 'Duracion': cuanto dura el bloqueo (solo para 'En un momento específico' y
        'Desde el inicio'; no aplica a 'Toda la planificación' ni a 'Entre Inicio y Fin').
      - 'Inicio': turno u hora en que arranca el bloqueo (para 'En un momento específico'
        y 'Entre Inicio y Fin'; con Unidad='Turnos', turno 1 = tiempo 0, misma convencion
        1-based que usa calcular_demoras_restricciones para las restricciones por turno).
      - 'Fin': turno u hora en que TERMINA el bloqueo, de forma inclusiva (solo para
        'Entre Inicio y Fin'; con Unidad='Turnos', el bloqueo cubre ese turno completo).

    Varias filas para el mismo frente se fusionan: si sus ventanas se solapan quedan
    como una sola ventana continua.
    """
    bloqueos_por_frente = {}
    for fila in bloqueos_config or []:
        frente = fila.get('Frente')
        if not frente:
            continue
        modo = fila.get('Modo', 'Toda la planificación')
        unidad = fila.get('Unidad', 'Horas')

        def _a_horas(valor):
            valor = float(valor or 0)
            return valor * duracion_turno if unidad == 'Turnos' else valor

        if modo == 'Toda la planificación':
            inicio_h, fin_h = 0.0, float(tiempo_limite)
        elif modo == 'Desde el inicio (N horas/turnos)':
            inicio_h = 0.0
            fin_h = min(_a_horas(fila.get('Duracion', 0)), float(tiempo_limite))
        elif modo == 'Entre Inicio y Fin (turno u hora)':
            valor_inicio = float(fila.get('Inicio', 1) or 1)
            valor_fin = float(fila.get('Fin', valor_inicio) or valor_inicio)
            if unidad == 'Turnos':
                # Turno 1 = tiempo 0; el 'Fin' es INCLUSIVO, así que un Fin=22 cubre el
                # turno 22 completo (igual convención que usa el motor de mantenciones
                # programadas para el "fin" de una ventana en turnos).
                inicio_h = max(0.0, (valor_inicio - 1) * duracion_turno)
                fin_h = max(inicio_h, valor_fin * duracion_turno)
            else:
                inicio_h = max(0.0, valor_inicio)
                fin_h = max(inicio_h, valor_fin)
            fin_h = min(fin_h, float(tiempo_limite))
        else:  # 'En un momento específico'
            valor_inicio = float(fila.get('Inicio', 1) or 1)
            if unidad == 'Turnos':
                # Turno 1 = tiempo 0 (misma convencion 1-based que el resto de la app)
                inicio_h = max(0.0, (valor_inicio - 1) * duracion_turno)
            else:
                inicio_h = max(0.0, valor_inicio)
            fin_h = min(inicio_h + _a_horas(fila.get('Duracion', 0)), float(tiempo_limite))

        if fin_h > inicio_h:
            bloqueos_por_frente.setdefault(frente, []).append((inicio_h, fin_h))

    return {
        frente: _fusionar_intervalos(intervalos)
        for frente, intervalos in bloqueos_por_frente.items()
    }


def resumen_bloqueos_tuneles(bloqueos_por_frente, duracion_turno):
    """Arma un DataFrame legible (una fila por ventana de bloqueo) para mostrarlo en
    el Menu 14 y confirmarle al usuario, ANTES de simular, exactamente que tuneles
    quedaran con avance 0 y durante que horas/turnos."""
    filas = []
    for frente, intervalos in bloqueos_por_frente.items():
        for inicio, fin in intervalos:
            turno_inicio = int(inicio / duracion_turno) + 1
            turno_fin = math.ceil(fin / duracion_turno)
            filas.append({
                'Frente': frente,
                'Desde (h)': round(inicio, 2),
                'Hasta (h)': round(fin, 2),
                'Duración (h)': round(fin - inicio, 2),
                'Turnos aprox.': f"{turno_inicio}-{turno_fin}" if turno_fin > turno_inicio else str(turno_inicio),
            })
    return pd.DataFrame(filas)


def sincronizar_inicio_fin_duracion_bloqueos(df_prev, df_edit):
    """Mantiene coherentes 'Inicio', 'Fin' y 'Duración' en la tabla de Bloqueo Total (Menú 14).
    Aplica a los modos 'En un momento específico' y 'Entre Inicio y Fin', que describen lo mismo:
      - Unidad='Turnos' (Fin INCLUSIVO): Duración = Fin - Inicio + 1   (turno 1 al 56 = 56 turnos).
      - Unidad='Horas':                  Duración = Fin - Inicio.
    La columna que el usuario acaba de editar manda: si cambia 'Fin' se recalcula 'Duración', si
    cambia 'Duración' se recalcula 'Fin'. Si cambia otra cosa (Modo, Inicio, Unidad) manda 'Fin' en
    'Entre Inicio y Fin' y 'Duración' en 'En un momento específico' (igual que usa el motor).
    Devuelve (df_normalizado, hubo_cambio). No toca 'Toda la planificación' ni 'Desde el inicio'."""
    df = df_edit.copy().reset_index(drop=True)
    for col in ('Inicio', 'Fin', 'Duración'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    prev = None
    if df_prev is not None and len(df_prev) == len(df):
        prev = df_prev.reset_index(drop=True)

    def _cambio(i, col):
        if prev is None or col not in prev.columns or col not in df.columns:
            return False
        a, b = df.at[i, col], pd.to_numeric(prev.at[i, col], errors='coerce')
        if pd.isna(a) or pd.isna(b):
            return bool(pd.isna(a) != pd.isna(b))
        return abs(float(a) - float(b)) > 1e-9

    hubo_cambio = False
    for i in df.index:
        modo = df.at[i, 'Modo'] if 'Modo' in df.columns else None
        if modo not in ('En un momento específico', 'Entre Inicio y Fin (turno u hora)'):
            continue
        ini, fin, dur = df.at[i, 'Inicio'], df.at[i, 'Fin'], df.at[i, 'Duración']
        if pd.isna(ini):
            continue
        unidad = df.at[i, 'Unidad'] if 'Unidad' in df.columns else None
        extra = 0.0 if unidad == 'Horas' else 1.0  # en turnos el Fin es inclusivo
        cambio_fin, cambio_dur = _cambio(i, 'Fin'), _cambio(i, 'Duración')
        if cambio_fin and not cambio_dur:
            manda_fin = True
        elif cambio_dur and not cambio_fin:
            manda_fin = False
        else:
            manda_fin = (modo == 'Entre Inicio y Fin (turno u hora)')
        if manda_fin:
            if pd.isna(fin):
                continue
            nuevo = max(float(fin) - float(ini) + extra, 0.0)
            if pd.isna(dur) or abs(float(dur) - nuevo) > 1e-9:
                df.at[i, 'Duración'] = nuevo
                hubo_cambio = True
        else:
            if pd.isna(dur) or float(dur) <= 0:
                continue
            nuevo = float(ini) + float(dur) - extra
            if pd.isna(fin) or abs(float(fin) - nuevo) > 1e-9:
                df.at[i, 'Fin'] = nuevo
                hubo_cambio = True
    return df, hubo_cambio


def _turnos_a_rangos_tuplas(turnos):
    """Igual que _formatear_turnos_como_rangos, pero retorna una lista de tuplas
    (turno_inicio, turno_fin) en vez de un string — para poder generar automaticamente
    filas de 'Bloqueo Total' (una fila por rango contiguo) a partir de los turnos donde
    la hoja Restriccion ya viene activa, sin tener que agregar turno por turno."""
    if not turnos:
        return []
    turnos_ordenados = sorted(set(int(t) for t in turnos))
    rangos = []
    inicio = anterior = turnos_ordenados[0]
    for t in turnos_ordenados[1:]:
        if t == anterior + 1:
            anterior = t
            continue
        rangos.append((inicio, anterior))
        inicio = anterior = t
    rangos.append((inicio, anterior))
    return rangos


def construir_estado_turnos_frente(frentes, n_turnos, duracion_turno, df_preview_rest, bloqueos_tuneles_dict):
    """
    Arma, para cada combinación (frente, turno), el estado que tendrá en la próxima
    simulación, para poder 'validarlo' visualmente turno a turno antes de simular:
      - 'Restricción (demora)': la hoja Restriccion (Polvorazo/PA FH/etc., que ya trae
        su propio 'Turno') queda 'Se activa' para ese frente en ese turno según la Vista
        Previa (Menú 14) — solo demora el ciclo, no lo bloquea.
      - 'Bloqueo total': el turno cae dentro de una ventana de Bloqueo Total (Menú 14).
      - 'Libre': ninguna de las anteriores.
    Nota: FH/ObrasCiviles no se incluyen acá porque su demora es constante por ciclo
    (no dependen de un turno puntual, ver calcular_demoras_restricciones); se muestran
    aparte, en el resumen de arriba.
    """
    estado = {frente: {t: 'Libre' for t in range(1, n_turnos + 1)} for frente in frentes}

    if df_preview_rest is not None and not df_preview_rest.empty and 'Fuente' in df_preview_rest.columns:
        df_r = df_preview_rest[
            df_preview_rest['Fuente'].astype(str).str.startswith('Restricción (')
            & (df_preview_rest['Estado'] == 'Se activa')
        ]
        for _, fila in df_r.iterrows():
            turno_val = fila.get('Turno')
            if pd.isna(turno_val):
                continue
            turno = int(turno_val)
            if not (1 <= turno <= n_turnos):
                continue
            afectados = [
                f.strip() for f in str(fila.get('Frentes Afectados', '')).split(',')
                if f.strip() and f.strip() != '—'
            ]
            for frente in afectados:
                if frente in estado:
                    estado[frente][turno] = 'Restricción (demora)'

    for frente, intervalos in (bloqueos_tuneles_dict or {}).items():
        if frente not in estado:
            continue
        for inicio, fin in intervalos:
            turno_ini = max(1, int(inicio / duracion_turno) + 1)
            turno_fin = min(n_turnos, math.ceil(fin / duracion_turno))
            for t in range(turno_ini, turno_fin + 1):
                estado[frente][t] = 'Bloqueo total'  # el bloqueo total prevalece sobre la demora

    return estado


def graficar_estado_turnos_bloqueo(estado_por_frente, n_turnos):
    """Grilla (heatmap) Frente x Turno para 'validar' de un vistazo qué turnos quedan
    con restricción (demora) o con bloqueo total configurado, antes de simular."""
    frentes_orden = list(estado_por_frente.keys())
    mapa_valor = {'Libre': 0, 'Restricción (demora)': 1, 'Bloqueo total': 2}
    turnos = list(range(1, n_turnos + 1))
    z = [[mapa_valor[estado_por_frente[f][t]] for t in turnos] for f in frentes_orden]
    texto = [[estado_por_frente[f][t] for t in turnos] for f in frentes_orden]
    fig = go.Figure(data=go.Heatmap(
        z=z, x=turnos, y=frentes_orden, text=texto,
        hovertemplate='Frente: %{y}<br>Turno: %{x}<br>Estado: %{text}<extra></extra>',
        colorscale=[
            [0.0, '#E0E0E0'], [0.34, '#E0E0E0'],
            [0.34, '#FFA726'], [0.67, '#FFA726'],
            [0.67, '#B71C1C'], [1.0, '#B71C1C'],
        ],
        zmin=0, zmax=2, showscale=False, xgap=1, ygap=1
    ))
    fig.update_layout(
        title="Validación por turno: gris = libre, naranja = restricción (demora), rojo = bloqueo total",
        xaxis_title="Turno", yaxis_title="Frente",
        height=max(220, 38 * len(frentes_orden) + 110),
        margin=dict(l=10, r=10, t=60, b=10)
    )
    return fig


def _formatear_turnos_como_rangos(turnos):
    """Convierte una lista de turnos (enteros) en un texto compacto de rangos,
    ej. [1,2,3,5,6,8] -> '1-3, 5-6, 8'. Mucho mas legible que listar turno por
    turno cuando hay decenas de turnos consecutivos con la misma restriccion
    (como en el input de ejemplo, donde 'PA FH' se repite turno a turno)."""
    if not turnos:
        return ""
    turnos_ordenados = sorted(set(int(t) for t in turnos))
    rangos = []
    inicio = anterior = turnos_ordenados[0]
    for t in turnos_ordenados[1:]:
        if t == anterior + 1:
            anterior = t
            continue
        rangos.append(f"{inicio}" if inicio == anterior else f"{inicio}-{anterior}")
        inicio = anterior = t
    rangos.append(f"{inicio}" if inicio == anterior else f"{inicio}-{anterior}")
    return ", ".join(rangos)


def analizar_restricciones_preview(df_frentes, df_fh, df_oc, df_res, radio_restriccion,
                                    radios_por_tipo, frentes_seleccionados):
    """
    Vista previa de restricciones — se puede llamar ANTES de simular (a diferencia
    de calcular_demoras_restricciones, que solo se corre al presionar SIMULAR).

    Recorre cada fila cargada en las hojas FH, ObrasCiviles y Restriccion y, con el
    radio de impacto y los frentes seleccionados ACTUALES, determina si esa fila
    efectivamente interseca a algun frente (queda 'Activa') o no (queda 'Inactiva').
    Asi el usuario ve de inmediato -sin correr nada- cuales restricciones van a
    generar demora y cuales quedan fuera de rango, en vez de tener que adivinarlo
    o descubrirlo recien en la traza de eventos despues de simular.

    Retorna (df_detalle, resumen):
      - df_detalle: DataFrame con una fila por cada restriccion cargada (Fuente,
        Tipo, Turno, X, Y, Radio Aplicado (m), Estado, Frentes Afectados).
      - resumen: dict con conteos agregados y la lista de turnos que quedan con
        al menos una restriccion activa.
    """
    filas = []
    frentes_a_revisar = df_frentes[df_frentes['Frentes'].isin(frentes_seleccionados)]
    tiene_index_frente = 'Index' in df_frentes.columns

    def _frentes_que_intersectan(Px, Py, radio):
        afectados = []
        for _, f in frentes_a_revisar.iterrows():
            if check_tunnel_restriction_intersection(Px, Py, f['Xi'], f['Yi'], f['Xf'], f['Yf'], radio):
                afectados.append(f['Frentes'])
        return afectados

    # FH (Frente de Hundimiento) y OC (Obras Civiles): demora constante por ciclo,
    # no dependen del turno.
    for df_puntos, fuente, tipo in ((df_fh, 'Frente Hundimiento (FH)', 'FH'),
                                     (df_oc, 'Obras Civiles (OC)', 'OC')):
        if df_puntos is not None and 'X' in df_puntos.columns and 'Y' in df_puntos.columns:
            for _, row in df_puntos.iterrows():
                Px, Py = row['X'], row['Y']
                afectados = _frentes_que_intersectan(Px, Py, radio_restriccion)
                filas.append({
                    'Fuente': fuente, 'Tipo': tipo, 'Turno': None, 'X': Px, 'Y': Py,
                    'Radio Aplicado (m)': radio_restriccion,
                    'Estado': 'Se activa' if afectados else 'No se activa',
                    'Frentes Afectados': ', '.join(afectados) if afectados else '—'
                })

    # Restriccion (R): Polvorazo, Polvorazo B, PA FH, etc. — demora especifica por turno.
    if df_res is not None and {'X', 'Y', 'Turno', 'Tipo'}.issubset(df_res.columns):
        tiene_index_i = 'Index_I' in df_res.columns
        for _, row in df_res.iterrows():
            Px, Py = row['X'], row['Y']
            tipo = str(row['Tipo'])
            turno = row['Turno']
            radio_tipo = restriction_radius(tipo, radio_restriccion, radios_por_tipo)

            index_i_valor = row.get('Index_I') if tiene_index_i else None
            if tiene_index_i and pd.notna(index_i_valor):
                # Referencia directa a un frente puntual (Index_I), sin depender del radio.
                afectados = []
                for _, f in frentes_a_revisar.iterrows():
                    idx_frente = f['Index'] if tiene_index_frente else None
                    try:
                        if idx_frente is not None and pd.notna(idx_frente) and \
                                int(float(index_i_valor)) == int(float(idx_frente)):
                            afectados.append(f['Frentes'])
                    except (ValueError, TypeError):
                        pass
            else:
                afectados = _frentes_que_intersectan(Px, Py, radio_tipo)

            filas.append({
                'Fuente': f'Restricción ({tipo})', 'Tipo': tipo, 'Turno': turno,
                'X': Px, 'Y': Py, 'Radio Aplicado (m)': radio_tipo,
                'Estado': 'Se activa' if afectados else 'No se activa',
                'Frentes Afectados': ', '.join(afectados) if afectados else '—'
            })

    df_detalle = pd.DataFrame(filas)
    if df_detalle.empty:
        return df_detalle, {'total': 0, 'activas': 0, 'inactivas': 0, 'por_tipo': {}, 'turnos_activos': []}

    activas_mask = df_detalle['Estado'] == 'Se activa'
    por_tipo = {
        tipo: f"{int((grupo['Estado'] == 'Se activa').sum())} de {len(grupo)} activas"
        for tipo, grupo in df_detalle.groupby('Tipo')
    }
    turnos_activos = sorted(set(
        int(t) for t in df_detalle.loc[activas_mask, 'Turno'].dropna()
    ))
    resumen = {
        'total': len(df_detalle),
        'activas': int(activas_mask.sum()),
        'inactivas': int((~activas_mask).sum()),
        'por_tipo': por_tipo,
        'turnos_activos': turnos_activos,
    }
    return df_detalle, resumen


def mostrar_resumen_restricciones_aplicadas(titulo_contexto="", con_detalle=True):
    """
    Muestra, en cualquier punto de la seccion de Resultados (Avance Final, Carta
    Gantt, Traza de Eventos, etc.), un resumen de que restricciones (FH, OC, R)
    efectivamente aplicaron demora en la ULTIMA simulacion corrida — para que el
    usuario no tenga que adivinar por que un tunel se demoro mas de lo esperado,
    ni bucear en la traza de eventos completa para encontrarlo.

    No hace nada (ni muestra nada) si todavia no se ha corrido ninguna simulacion.
    """
    bloqueos_totales_simulados = st.session_state.get('bloqueos_tuneles_dict_simulado', {}) or {}

    if ('demoras_constantes' not in st.session_state and 'demoras_por_turno' not in st.session_state
            and not bloqueos_totales_simulados):
        return

    demoras_constantes = st.session_state.get('demoras_constantes', {}) or {}
    demoras_por_turno = st.session_state.get('demoras_por_turno', []) or []
    frentes_con_fh_oc = {f: h for f, h in demoras_constantes.items() if h and h > 0}

    prefijo = f"**{titulo_contexto}** — " if titulo_contexto else ""

    if not frentes_con_fh_oc and not demoras_por_turno and not bloqueos_totales_simulados:
        st.info(
            f"{prefijo}En esta simulación no se activó ninguna restricción física "
            "(FH, ObrasCiviles, Restriccion o bloqueo total de túnel) para los frentes/turnos simulados."
        )
        return

    partes = []
    if frentes_con_fh_oc:
        detalle_fh_oc = ", ".join(f"{f} (+{r2(h)} h/ciclo)" for f, h in frentes_con_fh_oc.items())
        partes.append(f"Demora constante por FH/ObrasCiviles en: {detalle_fh_oc}.")
    if demoras_por_turno:
        turnos_con_restriccion = sorted(set(int(r['Turno']) for r in demoras_por_turno))
        rango_turnos = _formatear_turnos_como_rangos(turnos_con_restriccion)
        partes.append(
            f"Se activaron {len(demoras_por_turno)} restricciones por turno "
            f"(turnos: {rango_turnos})."
        )
    if bloqueos_totales_simulados:
        detalle_bloqueos = "; ".join(
            f"{f} ({', '.join(f'{r2(ini)}h–{r2(fin)}h' for ini, fin in intervalos)})"
            for f, intervalos in bloqueos_totales_simulados.items()
        )
        partes.append(
            f"🔒 {len(bloqueos_totales_simulados)} túnel(es) con BLOQUEO TOTAL (avance 0, sin "
            f"recursos asignados) en: {detalle_bloqueos}."
        )
    st.info(f"{prefijo}" + " ".join(partes))

    if con_detalle and demoras_por_turno:
        with st.expander("Ver detalle de restricciones activas en esta simulación"):
            mostrar_tabla(pd.DataFrame(demoras_por_turno), hide_index=True, width='stretch')

    if con_detalle and bloqueos_totales_simulados:
        with st.expander("Ver detalle de bloqueos totales de túnel activos en esta simulación"):
            duracion_turno_resumen = float(st.session_state.get('duracion_turno_simulada', 8.0))
            mostrar_tabla(
                resumen_bloqueos_tuneles(bloqueos_totales_simulados, duracion_turno_resumen),
                hide_index=True, width='stretch'
            )


def mostrar_resumen_demora_cambio_turno():
    """Resumen (Menú 17) de la demora programada por cambio de turno de la última simulación:
    cuántos cambios de turno hubo, demora sorteada promedio y horas efectivamente perdidas por túnel
    (promedio por simulación). Solo se muestra si estaba activada al simular."""
    df = st.session_state.get('df_traza_eventos_cache')
    if df is None or getattr(df, 'empty', True) or 'type' not in df.columns:
        return
    ventanas = df[df['type'] == 'shift_change_window']
    if ventanas.empty:
        return
    n_sims = max(1, df['Simulation_ID'].nunique())
    st.markdown("#### Demora programada por cambio de turno")
    c1, c2, c3 = st.columns(3)
    c1.metric("Cambios de turno por simulación", f"{len(ventanas) / n_sims:.1f}")
    c2.metric("Demora promedio por cambio (h)", f"{ventanas['Duration'].mean():.2f}")
    c3.metric("Demora total sorteada por simulación (h)", f"{ventanas['Duration'].sum() / n_sims:.1f}")
    ef = df[df['type'] == 'delay_shift_change']
    if ef.empty:
        st.caption("Ningún túnel quedó detenido por un cambio de turno (los equipos estaban libres en esos instantes).")
        return
    tabla = (ef.groupby('Resource_Origin_Tunnel')['Duration'].sum() / n_sims).rename('Horas detenido por cambio de turno (prom./sim.)')
    tabla = tabla.round(2).reset_index().rename(columns={'Resource_Origin_Tunnel': 'Túnel'})
    mostrar_tabla(tabla, hide_index=True, width='stretch')
    st.caption("Horas en que el avance de cada túnel quedó pausado por un cambio de turno. También aparecen como "
               "'Demora por Cambio de Turno' en la Carta Gantt, el Resumen de Tiempos, Monte Carlo, la traza y el Excel.")


def mostrar_restricciones_del_tunel(tunel):
    """
    Muestra, justo arriba del Resumen de Tiempos / Carta Gantt de UN túnel puntual
    (Menú 18), si ESE túnel en particular tuvo restricciones activas en la última
    simulación. El resumen global del Menú 17 (mostrar_resumen_restricciones_aplicadas)
    junta TODOS los túneles de la corrida; un túnel puntual puede perfectamente no
    haber sido afectado por ninguna restricción aunque la corrida completa sí haya
    activado varias en OTROS túneles — en ese caso es normal y esperado que su Carta
    Gantt no muestre barras rojas 'Demora por Restricción', y este aviso lo deja claro
    para que no se confunda con un error.
    """
    bloqueos_totales_simulados = st.session_state.get('bloqueos_tuneles_dict_simulado', {}) or {}
    intervalos_bloqueo_tunel = bloqueos_totales_simulados.get(tunel, [])

    if ('demoras_constantes' not in st.session_state and 'demoras_por_turno' not in st.session_state
            and not intervalos_bloqueo_tunel):
        return

    demoras_constantes = st.session_state.get('demoras_constantes', {}) or {}
    demoras_por_turno = st.session_state.get('demoras_por_turno', []) or []

    demora_constante_tunel = demoras_constantes.get(tunel, 0) or 0
    demoras_turno_tunel = [r for r in demoras_por_turno if r['Frente'] == tunel]

    if demora_constante_tunel <= 0 and not demoras_turno_tunel and not intervalos_bloqueo_tunel:
        st.caption(
            f"Este túnel ({tunel}) no tuvo restricciones activas en esta simulación: "
            "por eso su Carta Gantt no muestra barras rojas 'Demora por Restricción'."
        )
        return

    if intervalos_bloqueo_tunel:
        rangos_bloqueo = ", ".join(f"{r2(ini)}h–{r2(fin)}h" for ini, fin in intervalos_bloqueo_tunel)
        st.warning(
            f"🔒 Este túnel ({tunel}) tiene BLOQUEO TOTAL configurado en {rangos_bloqueo}: durante "
            "esa ventana su avance queda en 0 y no se le asignó ningún recurso (equivalente a "
            "'shift_State = False'). Debería verse como barra 'BLOQUEO TOTAL DE TÚNEL' en su Carta Gantt."
        )

    partes = []
    if demora_constante_tunel > 0:
        partes.append(f"demora constante por FH/ObrasCiviles de {r2(demora_constante_tunel)} h/ciclo")
    if demoras_turno_tunel:
        turnos_tunel = sorted(set(int(r['Turno']) for r in demoras_turno_tunel))
        partes.append(
            f"{len(demoras_turno_tunel)} restricción(es) por turno activas "
            f"(turnos: {_formatear_turnos_como_rangos(turnos_tunel)})"
        )
    if partes:
        st.info(
            f"Este túnel ({tunel}) tuvo " + " y ".join(partes) +
            ". Debería(n) verse como barra(s) roja(s) 'Demora por Restricción...' en su Carta Gantt."
        )


def _cambio_es_por_hora(cambio):
    """True si el cambio del plan de flota se definio por HORA (clave 'hora' con valor), y no por turno."""
    hora = cambio.get('hora')
    if hora is None:
        return False
    try:
        return not math.isnan(float(hora))
    except (TypeError, ValueError):
        return False


def inicio_cambio_flota_h(cambio, duracion_turno):
    """Hora absoluta (horas desde el inicio de la simulacion) en que arranca un cambio del plan de flota,
    SIN contar su demora. Si el cambio se definio por hora, es esa hora; si se definio por turno, es el
    inicio de ese turno: (turno - 1) * duracion_turno."""
    if _cambio_es_por_hora(cambio):
        return max(0.0, float(cambio['hora']))
    return max(0.0, float(cambio['turno']) - 1.0) * duracion_turno


def turno_de_hora(hora, duracion_turno):
    """Turno (1-based) al que pertenece una hora absoluta de la simulacion."""
    return int(max(0.0, hora) // duracion_turno) + 1


def eventos_plan_flota_ordenados(plan_recurso, recurso, duracion_turno):
    """Cambios del plan para un recurso, como lista [(hora_inicio, cambio)] en orden cronologico.
    Si varios cambios caen exactamente en el mismo instante (mismo turno, o la misma hora), se queda
    con el ultimo definido."""
    por_inicio = {}
    for c in plan_recurso:
        if c['recurso'] != recurso:
            continue
        por_inicio[round(inicio_cambio_flota_h(c, duracion_turno), 6)] = c
    return [(t, por_inicio[t]) for t in sorted(por_inicio)]


def calcular_ventanas_flota(recurso, cantidad_base, plan_recurso, duracion_turno):
    """
    A partir de la cantidad base de un recurso y su lista de cambios de plan
    (turno, cantidad_nueva, demora_horas), calcula las ventanas de actividad
    [activa_desde, activa_hasta) de cada instancia (unidad de equipo/cuadrilla).

    - Un aumento de cantidad crea instancias nuevas que se activan en
      'inicio_turno + demora_horas'.
    - Una disminucion de cantidad desactiva las instancias de mayor id en
      'inicio_turno + demora_horas'.
    - Los cambios se aplican en orden cronologico de turno; si hay varios
      cambios para el mismo recurso en el mismo turno, se aplica el ultimo
      definido en el plan.
    """
    # Cada cambio se puede definir por TURNO (empieza al inicio de ese turno) o por HORA (empieza en esa
    # hora absoluta de la simulacion). Si hay duplicados en el mismo instante, se queda con el ultimo.
    eventos_ordenados = eventos_plan_flota_ordenados(plan_recurso, recurso, duracion_turno)

    # IMPORTANTE: se usa math.ceil() y no int() para convertir cantidad -> N de instancias.
    # int() trunca (p.ej. int(0.5) = 0), lo que puede dejar un recurso con 0 instancias
    # cuando el Excel trae cantidades fraccionarias (ej. Roboshot = 0.5), bloqueando
    # permanentemente cualquier frente que dependa de ese recurso. math.ceil() redondea
    # hacia arriba (ceil(0.5) = 1), igual que ya hacia el number_input de la UI, pero
    # ahora queda blindado tambien para usos fuera de la interfaz (script/API/batch).
    cantidad_actual = math.ceil(cantidad_base)
    ventanas = [{'id': i, 'activa_desde': 0.0, 'activa_hasta': float('inf')} for i in range(cantidad_actual)]
    registro_cambios = []  # para trazabilidad en resultados / Gantt

    for inicio_h, ev in eventos_ordenados:
        tiempo_evento = inicio_h + ev.get('demora_horas', 0.0)
        por_hora_ev = _cambio_es_por_hora(ev)
        turno_ev = turno_de_hora(inicio_h, duracion_turno) if por_hora_ev else int(ev['turno'])
        cantidad_nueva = math.ceil(ev['cantidad_nueva'])

        if cantidad_nueva > cantidad_actual:
            # Agregar instancias nuevas, activas desde tiempo_evento
            for i in range(cantidad_actual, cantidad_nueva):
                ventanas.append({'id': i, 'activa_desde': tiempo_evento, 'activa_hasta': float('inf')})
            registro_cambios.append({
                'recurso': recurso, 'turno': turno_ev, 'hora_inicio': round(inicio_h, 4),
                'definido_por': 'Hora' if por_hora_ev else 'Turno', 'tipo': 'aumento',
                'cantidad_anterior': cantidad_actual, 'cantidad_nueva': cantidad_nueva,
                'demora_horas': ev.get('demora_horas', 0.0), 'tiempo_efectivo': tiempo_evento
            })
        elif cantidad_nueva < cantidad_actual:
            # Cerrar la ventana de las instancias de mayor id (las ultimas agregadas)
            ids_a_cerrar = sorted([v['id'] for v in ventanas if v['activa_hasta'] == float('inf')], reverse=True)[:cantidad_actual - cantidad_nueva]
            for v in ventanas:
                if v['id'] in ids_a_cerrar:
                    v['activa_hasta'] = tiempo_evento
            registro_cambios.append({
                'recurso': recurso, 'turno': turno_ev, 'hora_inicio': round(inicio_h, 4),
                'definido_por': 'Hora' if por_hora_ev else 'Turno', 'tipo': 'disminucion',
                'cantidad_anterior': cantidad_actual, 'cantidad_nueva': cantidad_nueva,
                'demora_horas': ev.get('demora_horas', 0.0), 'tiempo_efectivo': tiempo_evento
            })
        cantidad_actual = cantidad_nueva

    return ventanas, registro_cambios


# ----------------------------------------------------------------------------------
# ---- Mantenciones PROGRAMADAS de equipos (Menu 12) ----
# ----------------------------------------------------------------------------------
# Distinto de 'fallas_equipos' (Menu 11): una falla es un evento ALEATORIO e imprevisto
# que se juega por CICLO para un frente/recurso y solo agrega una demora puntual al
# ciclo. Una mantencion PROGRAMADA en cambio es un atributo del EQUIPO (una instancia
# especifica, ej. "Jumbo_A #1"): mientras dura, esa unidad queda fisicamente NO
# disponible para NINGUN frente (igual que 'activa_desde'/'activa_hasta' del plan de
# flota), y el motor de asignacion de recursos ya elige la instancia libre mas
# conveniente entre todas las del mismo tipo, asi que si hay mas de una unidad el
# resto sigue trabajando con normalidad; si solo hay una (como en el ejemplo del
# usuario), el frente que la necesita simplemente espera a que la mantencion termine.
# Ambos mecanismos pueden coexistir sin pisarse: un equipo puede fallar de forma
# imprevista (Menu 11) aunque tenga o no una mantencion programada (Menu 12) en curso o
# futura; no comparten estructura de datos ni se descuentan entre si.
def _tiempo_disponible_tras_mantenciones(tiempo, ventanas_mantenimiento):
    """Dado un instante candidato en que una instancia de equipo podria empezar a
    trabajar, lo empuja hacia adelante si cae dentro de alguna ventana de mantencion
    programada de esa misma instancia (lista de tuplas (inicio_h, fin_h)). Repite el
    chequeo por si el nuevo instante cae dentro de OTRA ventana (mantenciones
    consecutivas o reintentos por falla de la propia mantencion)."""
    if not ventanas_mantenimiento:
        return tiempo
    cambiado = True
    while cambiado:
        cambiado = False
        for inicio, fin in ventanas_mantenimiento:
            if inicio <= tiempo < fin:
                tiempo = fin
                cambiado = True
    return tiempo


def _fin_con_mantenciones(inicio, duracion, ventanas_mantenimiento):
    """Instante en que termina una actividad de 'duracion' horas de trabajo efectivo que empieza
    en 'inicio' (ya fuera de mantencion) sobre una instancia con ventanas de mantencion
    (inicio_h, fin_h). Si una ventana cae a mitad de la actividad, el equipo se retira: el trabajo
    se PAUSA durante toda la ventana y se reanuda al terminar (el tiempo de mantencion no cuenta
    como trabajo). Sin ventanas, es simplemente inicio + duracion."""
    if not ventanas_mantenimiento or duracion <= 0:
        return inicio + duracion
    t = inicio
    restante = duracion
    for v_ini, v_fin in sorted(ventanas_mantenimiento):
        if v_fin <= t:
            continue                    # ventana ya pasada
        if v_ini >= t + restante:
            break                       # la actividad termina antes de que empiece la ventana
        # la ventana empieza a mitad de la actividad (v_ini > t; si v_ini <= t ya se empujo el inicio)
        if v_ini > t:
            restante -= (v_ini - t)
        t = max(t, v_fin)
    return t + restante


def _pausas_por_mantencion(inicio, duracion, ventanas_mantenimiento):
    """Mismos supuestos que _fin_con_mantenciones, pero devuelve las PAUSAS (inicio_h, fin_h)
    que sufre una actividad de 'duracion' horas de trabajo efectivo que parte en 'inicio'
    porque una ventana de mantención cae a mitad de ella. Sirve para dejar cada pausa
    registrada como evento propio en la traza (trazabilidad: nada de tiempo oculto)."""
    pausas = []
    if not ventanas_mantenimiento or duracion <= 0:
        return pausas
    t = inicio
    restante = duracion
    for v_ini, v_fin in sorted(ventanas_mantenimiento):
        if v_fin <= t:
            continue
        if v_ini >= t + restante:
            break
        if v_ini > t:
            restante -= (v_ini - t)
            pausas.append((v_ini, v_fin))
        else:
            pausas.append((t, v_fin))
        t = max(t, v_fin)
    return pausas


NOMBRE_CAMBIO_TURNO = "Cambio de turno"   # centinela: ventana de 'Demora programada por cambio de turno' (Menú 5)


def generar_ventanas_cambio_turno(config, duracion_turno, tiempo_limite):
    """Ventanas (inicio_h, fin_h) de demora por CAMBIO DE TURNO para UNA simulación. En cada fin de
    turno (k * duracion_turno, con k = 1, 2, ... antes del horizonte) el sistema se detiene una demora
    sorteada de la distribución elegida en el Menú 5. La demora se recorta a la duración del turno y al
    horizonte simulado. Sin configuración activa devuelve []."""
    if not config or not config.get('activo') or duracion_turno <= 0:
        return []
    ventanas = []
    k = 1
    while k * duracion_turno < tiempo_limite:
        inicio = k * duracion_turno
        d = float(generar_tiempos_batch(config['params'], config['distribucion'], 1)[0])
        d = max(0.0, min(d, duracion_turno))
        fin = min(tiempo_limite, inicio + d)
        if fin - inicio > 1e-9:
            ventanas.append((inicio, fin))
        k += 1
    return ventanas


def _nombres_mantencion_en_rango(inst, t0, t1):
    """Nombres de las mantenciones programadas de la instancia que se solapan con [t0, t1]."""
    nombres = []
    for info in inst.get('mantenciones_info', []):
        if info['ini'] < t1 and info['fin'] > t0 and info['nombre'] not in nombres:
            nombres.append(info['nombre'])
    return nombres


def _eventos_demora_mantencion(sim, ciclo_global, ciclo_frente, secuencia, frente, recurso, inst,
                               intervalos, tiempo_limite, pos_x, pos_y, vistos_turno=None):
    """Eventos de traza 'delay_maintenance': tiempo que un FRENTE/avance concreto tuvo que
    esperar (o que su actividad quedó pausada) porque el equipo que necesitaba estaba en
    mantención programada (Menú 12). Van asociados al túnel y al avance, así aparecen en la
    Carta Gantt, el Resumen de Tiempos, la distribución Monte Carlo y la Traza por avance.
    Si la pausa corresponde SOLO a una ventana de cambio de turno (Menú 5), el evento sale como
    'delay_shift_change' (una sola vez por túnel y ventana, aunque varios equipos esperen)."""
    eventos = []
    for a, b in intervalos:
        b = min(b, tiempo_limite)
        if b - a <= 1e-6:
            continue
        nombres = _nombres_mantencion_en_rango(inst, a, b)
        if nombres and all(n == NOMBRE_CAMBIO_TURNO for n in nombres):
            clave = (frente, round(a, 4), round(b, 4))
            if vistos_turno is not None:
                if clave in vistos_turno:
                    continue
                vistos_turno.add(clave)
            # Ventana COMPLETA sorteada de la distribucion (Menu 5). 'Duration' puede ser menor porque el
            # equipo solo espera el resto de la ventana si llega a mitad de ella; para comparar contra la
            # distribucion de entrada (y su rango truncado) se usa la ventana completa.
            _v_ini, _v_fin = a, b
            for _info in inst.get('mantenciones_info', []):
                if _info['nombre'] == NOMBRE_CAMBIO_TURNO and _info['ini'] < b and _info['fin'] > a:
                    _v_ini, _v_fin = _info['ini'], _info['fin']
                    break
            eventos.append({
                'Simulation_ID': sim + 1, 'Cycle': ciclo_global, 'Ciclo_Frente': ciclo_frente,
                'Secuencia': secuencia,
                'Actividad': "Demora por Cambio de Turno",
                'Recurso': 'N/A', 'type': 'delay_shift_change',
                'Start': round(a, 2), 'Finish': round(b, 2), 'Duration': round(b - a, 2),
                'Duration_Muestreada': round(_v_fin - _v_ini, 4), 'Ventana_Inicio': round(_v_ini, 4),
                'Start_X_Front': round(pos_x, 2), 'Start_Y_Front': round(pos_y, 2),
                'End_X_Front': round(pos_x, 2), 'End_Y_Front': round(pos_y, 2),
                'Cycle_Advance_Length_m': 0, 'Resource_Origin_Tunnel': frente,
                'Resource_Destination_Tunnel': 'N/A', 'Travel_Time_Actual': 0,
                'Travel_Speed_Used_m_h': 0, 'Travel_Distance_m': 0,
                'Es_Personalizada': False
            })
            continue
        nombres = [n for n in nombres if n != NOMBRE_CAMBIO_TURNO]
        eventos.append({
            'Simulation_ID': sim + 1, 'Cycle': ciclo_global, 'Ciclo_Frente': ciclo_frente,
            'Secuencia': secuencia,
            'Actividad': f"Demora por Mantención Programada ({recurso} #{inst['id'] + 1})",
            'Recurso': f"{recurso}_{inst['id']}", 'type': 'delay_maintenance',
            'Detalle_Mantencion': ", ".join(nombres),
            'Start': round(a, 2), 'Finish': round(b, 2), 'Duration': round(b - a, 2),
            'Start_X_Front': round(pos_x, 2), 'Start_Y_Front': round(pos_y, 2),
            'End_X_Front': round(pos_x, 2), 'End_Y_Front': round(pos_y, 2),
            'Cycle_Advance_Length_m': 0, 'Resource_Origin_Tunnel': frente,
            'Resource_Destination_Tunnel': 'N/A', 'Travel_Time_Actual': 0,
            'Travel_Speed_Used_m_h': 0, 'Travel_Distance_m': 0,
            'Es_Personalizada': False
        })
    return eventos


def generar_ventanas_mantenimiento_instancia(mantencion, duracion_turno, tiempo_limite):
    """Genera, PARA UNA simulacion (se debe llamar de nuevo en cada 'sim' del loop de
    Monte Carlo), la o las ventanas de indisponibilidad que produce un escenario de
    mantencion programada sobre la instancia de equipo que describe.

    - 'modo_tiempo': 'Turnos' u 'Horas' — en que unidad vienen 'inicio'/'fin'.
    - 'modo_duracion': 'Manual (inicio-fin)' -> la duracion es fija, fin - inicio.
                        'Distribución estadística' -> la duracion se muestrea cada
                        intento con generar_tiempos_batch (misma logica que Menu 8),
                        pudiendo ser tan simple como 'Cte' (duracion fija con forma
                        de distribucion) o cualquiera de las distribuciones de
                        DISTRIBUTION_PARAM_INFO.
    - 'probabilidad_exito': % de que la mantencion termine bien. Si fracasa, el equipo
      "vuelve a mantencion" de inmediato (nueva ventana pegada a continuacion de la
      anterior) hasta que una intentona tenga exito o se acabe el horizonte simulado.

    Devuelve (ventanas, eventos): 'ventanas' es la lista de (inicio_h, fin_h) para
    aplicar sobre la instancia; 'eventos' trae el detalle de cada intento (exito o
    fracaso) listo para volcarse a traza_eventos."""
    modo_tiempo = mantencion.get('modo_tiempo', 'Horas')

    def _a_horas(valor, es_fin=False):
        if modo_tiempo == 'Turnos':
            base = (float(valor) - 1) * duracion_turno
            # El "fin" de un turno es el CIERRE de ese turno, no su apertura (turno 1
            # = [0, duracion_turno), igual convencion que el resto del motor).
            return base + duracion_turno if es_fin else base
        return float(valor)

    inicio_h = max(0.0, min(_a_horas(mantencion['inicio']), tiempo_limite))
    modo_duracion = mantencion.get('modo_duracion', 'Manual (inicio-fin)')
    # En modo 'Turnos' la mantención ocupa turnos ENTEROS (ej. turnos 2 a 6 = turnos 2,3,4,5,6
    # completos): nunca hay distribución de duración. Solo en modo 'Horas' (hora de inicio +
    # fin manual o duración por distribución, ej. inicio hora 2 + Gauss -> 2.6 h => 2 a 4.6).
    if modo_tiempo == 'Turnos':
        modo_duracion = 'Manual (inicio-fin)'
    prob_exito = float(mantencion.get('probabilidad_exito', 100))
    # Si el usuario definió una distribución PROPIA para el tiempo de reintento (Menú 12),
    # esta se usa solo desde el 2do intento en adelante; el 1er intento SIEMPRE usa la
    # duración/distribución de la mantención original.
    modo_reintento = mantencion.get('modo_reintento', 'Igual a la mantención original')
    if modo_tiempo == 'Turnos':
        modo_reintento = 'Igual a la mantención original'

    duracion_manual = None
    if modo_duracion != 'Distribución estadística':
        duracion_manual = max(0.0, _a_horas(mantencion['fin'], es_fin=True) - inicio_h)

    def _duracion_intento(intento_actual):
        if intento_actual > 1 and modo_reintento == 'Definir una distribución propia para el reintento':
            return float(generar_tiempos_batch(
                mantencion['params_distribucion_reintento'], mantencion['distribucion_reintento'], 1
            )[0])
        if modo_duracion == 'Distribución estadística':
            return float(generar_tiempos_batch(
                mantencion['params_distribucion'], mantencion['distribucion'], 1
            )[0])
        return duracion_manual

    ventanas, eventos = [], []
    t_actual = inicio_h
    intento = 0
    MAX_INTENTOS = 20  # blindaje anti-loop-infinito si probabilidad_exito quedara en 0%

    while intento < MAX_INTENTOS and t_actual < tiempo_limite:
        intento += 1
        duracion = max(0.0, _duracion_intento(intento))
        fin_actual = min(tiempo_limite, t_actual + duracion)
        exito = (np.random.rand() * 100.0) < prob_exito

        ventanas.append((t_actual, fin_actual))
        eventos.append({
            'Start': t_actual, 'Finish': fin_actual, 'Duration': fin_actual - t_actual,
            'Intento': intento, 'Exito': exito,
            'Duracion_Muestreada': duracion   # duracion sorteada completa (antes de cortar por el horizonte)
        })

        if exito or fin_actual >= tiempo_limite:
            break
        t_actual = fin_actual  # fracasó: el equipo vuelve a mantención de inmediato (reintento)

    return ventanas, eventos


def construir_calendario_mantenciones(mantenciones, duracion_turno, n_turnos):
    """Arma, para cada equipo (Recurso + N° de instancia), el estado 'Libre' /
    'Mantención programada' turno a turno, para VALIDAR VISUALMENTE la configuración
    antes de simular (calendario, igual espíritu que la Vista Previa del Menú 14).

    Usa una duración REPRESENTATIVA solo para este dibujo (el fin manual si el modo es
    manual, o el promedio de la distribución si el modo es 'Distribución estadística');
    la simulación real vuelve a sortear duración y éxito de forma independiente en
    cada corrida Monte Carlo, así que turnos puntuales pueden variar corrida a corrida."""
    equipos = sorted({f"{m['recurso']} #{m['instancia_id'] + 1}" for m in mantenciones})
    estado = {eq: {t: 'Libre' for t in range(1, n_turnos + 1)} for eq in equipos}

    for m in mantenciones:
        eq = f"{m['recurso']} #{m['instancia_id'] + 1}"
        modo_tiempo = m.get('modo_tiempo', 'Horas')

        def _a_horas(v, es_fin=False):
            if modo_tiempo == 'Turnos':
                base = (float(v) - 1) * duracion_turno
                return base + duracion_turno if es_fin else base
            return float(v)

        inicio_h = _a_horas(m['inicio'])
        if m.get('modo_duracion') == 'Distribución estadística' and modo_tiempo != 'Turnos':
            duracion_rep = 0.0
            try:
                dist_obj = DIST_OBJ_SCIPY.get(m['distribucion'])
                if dist_obj is not None:
                    kwargs = _kwargs_scipy_desde_params(m['params_distribucion'], m['distribucion'])
                    valor_medio = dist_obj.mean(**kwargs)
                    if np.isfinite(valor_medio):
                        duracion_rep = float(valor_medio)
                elif m['distribucion'] == 'Cte':
                    duracion_rep = float(m['params_distribucion'])
            except Exception:
                duracion_rep = 0.0
            fin_h = inicio_h + max(0.0, duracion_rep)
        else:
            fin_h = _a_horas(m['fin'], es_fin=True)

        turno_ini = max(1, int(inicio_h / duracion_turno) + 1)
        turno_fin = min(n_turnos, math.ceil(fin_h / duracion_turno))
        for t in range(turno_ini, turno_fin + 1):
            if eq in estado:
                estado[eq][t] = 'Mantención programada'

    return estado


def graficar_calendario_mantenciones(estado_por_equipo, n_turnos):
    """Grilla (heatmap) Equipo x Turno para validar de un vistazo qué turnos quedan
    con mantención programada, antes de simular (mismo estilo que
    graficar_estado_turnos_bloqueo, del Menú 14)."""
    equipos_orden = list(estado_por_equipo.keys())
    mapa_valor = {'Libre': 0, 'Mantención programada': 1}
    turnos = list(range(1, n_turnos + 1))
    z = [[mapa_valor[estado_por_equipo[e][t]] for t in turnos] for e in equipos_orden]
    texto = [[estado_por_equipo[e][t] for t in turnos] for e in equipos_orden]
    fig = go.Figure(data=go.Heatmap(
        z=z, x=turnos, y=equipos_orden, text=texto,
        hovertemplate='Equipo: %{y}<br>Turno: %{x}<br>Estado: %{text}<extra></extra>',
        colorscale=[[0.0, '#E0E0E0'], [0.5, '#E0E0E0'], [0.5, '#1976D2'], [1.0, '#1976D2']],
        zmin=0, zmax=1, showscale=False, xgap=1, ygap=1
    ))
    fig.update_layout(
        title="Calendario de mantenciones: gris = equipo disponible, azul = mantención programada",
        xaxis_title="Turno", yaxis_title="Equipo",
        height=max(220, 38 * len(equipos_orden) + 110),
        margin=dict(l=10, r=10, t=60, b=10)
    )
    return fig


# ======================================================================
# GEOLOGIA DESDE MWD HISTORICO (opcion adicional a los ponderadores)
# ----------------------------------------------------------------------
# El MWD solo clasifica frentes historicos por ROP medio (P20/P40/P60/P80) y entrega,
# por clase de roca, un catalogo de referencias empiricas. La clase de cada tunel la
# asigna el usuario. En cada replica Monte Carlo, cada tunel sortea una referencia MWD
# de su clase por bloque de N ciclos; el percentil de ROP de esa referencia fija el
# percentil de la distribucion de tiempo de las actividades con regla directa/inversa.
# La etiqueta de roca es RELATIVA a la base MWD cargada, y la asociacion percentil de
# ROP -> percentil de tiempo es una regla de escenario, no una relacion calibrada.
# ======================================================================
MWD_CLASES = ["Muy buena", "Buena", "Regular", "Mala", "Muy mala"]
MWD_COLORES = {"Muy buena": "#2e7d32", "Buena": "#9ccc65", "Regular": "#fdd835",
               "Mala": "#fb8c00", "Muy mala": "#c62828"}
MWD_COLUMNAS = ["Frente_Trabajo", "ID_Barreno", "ROP_Normalizado_Final"]
MWD_MODOS = ["Fija", "Aleatoria", "Sin MWD"]
MWD_REGLAS = ["libre", "directa", "inversa"]


def mwd_reglas_jumbo(actividades_nivel):
    """Reglas automaticas: toda actividad cuyo recurso es un jumbo (perforacion) usa la regla inversa
    (mas ROP, menos horas). Se detecta por el nombre del recurso, sin intervencion del usuario."""
    try:
        df = actividades_nivel[['Actividad', 'Recurso']].dropna(subset=['Actividad'])
        mask = df['Recurso'].astype(str).str.contains('jumbo', case=False, na=False)
        return {str(a): 'inversa' for a in df.loc[mask, 'Actividad'].astype(str).unique()}
    except Exception:
        return {}


def _mwd_abrir(fuente):
    """Ruta o archivo subido -> objeto binario posicionado al inicio."""
    if isinstance(fuente, (str, os.PathLike)):
        return open(fuente, 'rb')
    fuente.seek(0)
    return fuente


def _mwd_formato(fuente):
    """Detecta la fila de cabecera (el archivo puede traer lineas de titulo antes) y el separador."""
    f = _mwd_abrir(fuente)
    fila_cab, cab = None, None
    try:
        for i, linea in enumerate(f):
            texto = linea.decode('utf-8-sig', errors='replace') if isinstance(linea, bytes) else linea
            if 'Frente_Trabajo' in texto:
                fila_cab, cab = i, texto.rstrip('\r\n')
                break
            if i > 200:
                break
    finally:
        if isinstance(fuente, (str, os.PathLike)):
            f.close()
        else:
            fuente.seek(0)
    if fila_cab is None:
        raise ValueError("No se encontró la columna 'Frente_Trabajo' en las primeras 200 líneas del archivo.")
    sep = max([";", ",", "\t", "|"], key=cab.count)
    columnas = [c.strip() for c in cab.split(sep)]
    faltan = [c for c in MWD_COLUMNAS if c not in columnas]
    if faltan:
        raise ValueError(f"Faltan columnas MWD: {faltan}. Columnas encontradas: {columnas}")
    return fila_cab, sep


def mwd_vista_previa(fuente, n=100):
    """Primeras n filas del CSV tal como vienen (texto) y sus columnas, para mostrar al usuario que se leyo."""
    try:
        fila_cab, sep = _mwd_formato(fuente)
        src = fuente if isinstance(fuente, (str, os.PathLike)) else _mwd_abrir(fuente)
        df = pd.read_csv(src, sep=sep, encoding='utf-8-sig', skiprows=fila_cab, nrows=n, index_col=False, dtype=str, engine='c')
        if not isinstance(fuente, (str, os.PathLike)):
            fuente.seek(0)
        return {'vista_previa': df, 'columnas': list(df.columns), 'fila_cab': int(fila_cab), 'sep': sep}
    except Exception as e:
        return {'vista_previa': None, 'columnas': [], 'error_vista': str(e)}


_MWD_USO_COLUMNA = {
    'Frente_Trabajo': "SE USA. Identifica el frente: todas las filas con el mismo nombre se agrupan en un frente.",
    'ID_Barreno': "SE USA. Identifica el barreno dentro del frente: cada barreno pesa igual al promediar.",
    'ROP_Normalizado_Final': "SE USA. Es la variable que se promedia (primero por barreno, luego por frente) y se clasifica.",
}


def _mwd_ui_explicacion(cat, info):
    """Explica al usuario que se hizo con el CSV: columnas, primeros registros, pasos, resultado y un ejemplo."""
    prev = info.get('vista_previa')
    with st.expander("Qué se hizo con el archivo cargado (columnas, pasos y resultado)"):
        t1, t2, t3 = st.tabs(["1. Columnas y primeros 100 registros", "2. Qué se hizo, paso a paso", "3. Ejemplo con un frente"])
        with t1:
            if prev is None or len(prev) == 0:
                st.caption("No hay vista previa disponible para este archivo. Vuelva a presionar 'Procesar MWD'.")
            else:
                cols = list(prev.columns)
                st.caption(f"Archivo: {info.get('origen', '')} | cabecera en la línea {info.get('fila_cab', 0) + 1} | separador "
                           f"{info.get('sep', ';')!r} | {len(cols)} columnas. Solo 3 se usan; las demás se ignoran al leer "
                           "(así el archivo grande se procesa más rápido y con menos memoria).")
                filas_c = [{'Columna': c, 'Ejemplo (primera fila)': str(prev[c].iloc[0]),
                            'Uso': _MWD_USO_COLUMNA.get(c, "No se usa en la clasificación.")} for c in cols]
                mostrar_tabla(pd.DataFrame(filas_c), width='stretch', hide_index=True)
                st.markdown(f"**Primeros {len(prev)} registros del archivo (sin modificar)**")
                mostrar_tabla(prev, width='stretch', hide_index=True)
        with t2:
            cortes = info.get('cortes') or [float(v) for v in cat['rop_medio'].quantile([0.2, 0.4, 0.6, 0.8])]
            filas, exc = info.get('filas', 0), info.get('excluidas', 0)
            st.markdown(
                f"1. **Lectura.** Se leyeron **{filas:,}** filas del CSV por bloques (sin cargarlo completo en memoria); "
                f"**{exc:,}** se excluyeron por no tener frente, barreno o ROP numérico válido.\n"
                f"2. **Promedio por barreno.** Las filas de cada barreno (una por profundidad medida) se promedian en un solo "
                f"valor de `ROP_Normalizado_Final`. Hay **{int(cat['n_barrenos'].sum()):,}** barrenos.\n"
                f"3. **Promedio por frente.** Los barrenos de cada frente se promedian con igual peso: eso es `rop_medio`. "
                f"Quedan **{len(cat):,}** frentes, una fila por frente, y `n_barrenos` indica cuántos barrenos tiene cada uno.\n"
                f"4. **Cortes de clase.** Con los `rop_medio` de todos los frentes se calculan P20 = **{cortes[0]:.4f}**, "
                f"P40 = **{cortes[1]:.4f}**, P60 = **{cortes[2]:.4f}** y P80 = **{cortes[3]:.4f}**.\n"
                f"5. **Clase de roca.** ROP <= P20: Muy buena; hasta P40: Buena; hasta P60: Regular; hasta P80: Mala; sobre P80: "
                f"Muy mala. ROP bajo significa roca más dura (muy buena).\n"
                f"6. **p_rop.** Es la posición del frente entre todos, de 0 a 1: (ranking - 0.5) / {len(cat):,}. Frentes con el mismo "
                f"ROP medio comparten el mismo p_rop y la misma clase.")
            st.caption("Las clases son relativas a esta base: cambian si cambia el archivo. Si alguna clase queda vacía "
                       "es porque muchos frentes empatan en un mismo valor.")
            if filas in (999999, 1000000, 1048575, 1048576):
                st.warning(f"El archivo tiene exactamente {filas:,} filas, un número típico del límite de una planilla o de una "
                           "exportación cortada. Confirme que el CSV no esté truncado.")
            dup = cat.duplicated(subset=['rop_medio', 'n_barrenos'], keep=False)
            if dup.any():
                st.warning(f"{int(dup.sum())} frentes comparten exactamente el mismo ROP medio y número de barrenos con otro frente. "
                           "Puede ser casualidad o mediciones duplicadas con otro nombre; revise el CSV de origen.")
        with t3:
            nombres = list(cat.sort_values('rop_medio')['Frente_Trabajo'].astype(str))
            f_sel = st.selectbox("Frente a explicar", nombres, index=len(nombres) // 2, key="mwd_ej_frente")
            fila = cat[cat['Frente_Trabajo'].astype(str) == f_sel].iloc[0]
            rk = float(cat['rop_medio'].rank(method='average')[cat['Frente_Trabajo'].astype(str) == f_sel].iloc[0])
            cortes = [float(v) for v in cat['rop_medio'].quantile([0.2, 0.4, 0.6, 0.8])]
            st.markdown(
                f"- El frente tiene **{int(fila['n_barrenos'])}** barrenos; el promedio de los promedios de sus barrenos es "
                f"`rop_medio` = **{fila['rop_medio']:.4f}**.\n"
                f"- Ordenado de menor a mayor ROP entre los {len(cat):,} frentes, queda en la posición **{rk:g}**: "
                f"p_rop = ({rk:g} - 0.5) / {len(cat):,} = **{fila['p_rop']:.4f}**.\n"
                f"- Se compara con los cortes (P20 {cortes[0]:.4f} | P40 {cortes[1]:.4f} | P60 {cortes[2]:.4f} | P80 {cortes[3]:.4f}) "
                f"y resulta clase **{fila['calidad_roca']}**.")


def mwd_clasificar(resumen):
    """Entra: una fila por frente con 'rop_medio'. Sale: mismo DataFrame con calidad_roca y p_rop.

    Misma metodologia acordada: P20/P40/P60/P80 (interpolacion lineal de pandas) del ROP medio por
    frente; ROP <= P20 Muy buena ... ROP > P80 Muy mala; los empates reciben la misma categoria.
    p_rop = (rank - 0.5) / N (posicion media: evita P0/P100 y da igual percentil a valores iguales)."""
    if len(resumen) < 2 or resumen['rop_medio'].nunique() < 2:
        raise ValueError("Se necesitan al menos dos frentes con ROP medio distinto.")
    rop = resumen['rop_medio']
    cortes = rop.quantile([0.2, 0.4, 0.6, 0.8]).to_numpy()
    clases = np.searchsorted(cortes, rop.to_numpy(), side='left')
    resumen = resumen.copy()
    resumen['calidad_roca'] = np.array(MWD_CLASES)[clases]
    resumen['p_rop'] = (rop.rank(method='average') - 0.5) / len(rop)
    return resumen


def mwd_catalogo_desde_csv(fuente, chunksize=1_000_000, log=None):
    """Lee un CSV MWD (ruta o archivo subido) por bloques, sin cargarlo completo en memoria.

    Acumula suma y conteo del ROP por (frente, barreno); la media por barreno resultante es
    identica a promediar el CSV completo, asi que no se pierde informacion. Luego promedia los
    barrenos de cada frente con igual peso y clasifica. Entra: ruta o archivo. Sale: (catalogo, info)."""
    log = log or (lambda m: None)
    t0 = time.time()
    fila_cab, sep = _mwd_formato(fuente)
    log(f"MWD: cabecera en línea {fila_cab}, separador {sep!r}")
    kw = dict(sep=sep, encoding='utf-8-sig', usecols=MWD_COLUMNAS, index_col=False,
              skiprows=fila_cab, dtype={'Frente_Trabajo': str, 'ID_Barreno': str}, engine='c')

    def _src():
        return fuente if isinstance(fuente, (str, os.PathLike)) else _mwd_abrir(fuente)

    prueba = pd.read_csv(_src(), nrows=2000, decimal='.', **kw)
    decimal = '.' if pd.api.types.is_numeric_dtype(prueba['ROP_Normalizado_Final']) else ','
    log(f"MWD: decimal detectado {decimal!r}")

    acum, n_filas, n_excl, n_chunk = None, 0, 0, 0
    for chunk in pd.read_csv(_src(), decimal=decimal, chunksize=chunksize, **kw):
        n_chunk += 1
        n_filas += len(chunk)
        rop = pd.to_numeric(chunk['ROP_Normalizado_Final'], errors='coerce')
        valido = chunk['Frente_Trabajo'].notna() & chunk['ID_Barreno'].notna() & np.isfinite(rop)
        n_excl += int((~valido).sum())
        c = pd.DataFrame({'f': chunk.loc[valido, 'Frente_Trabajo'], 'b': chunk.loc[valido, 'ID_Barreno'],
                          'r': rop[valido]})
        parcial = c.groupby(['f', 'b'], sort=False)['r'].agg(['sum', 'count'])
        acum = parcial if acum is None else pd.concat([acum, parcial]).groupby(level=[0, 1], sort=False).sum()
        log(f"MWD: bloque {n_chunk} procesado, {n_filas:,} filas leídas, {len(acum):,} barrenos acumulados")
    if acum is None or acum.empty:
        raise ValueError("El archivo no tiene filas válidas de ROP.")
    barrenos = acum['sum'] / acum['count']
    resumen = barrenos.groupby(level=0).mean().rename('rop_medio').reset_index().rename(columns={'f': 'Frente_Trabajo'})
    resumen['n_barrenos'] = barrenos.groupby(level=0).size().reindex(resumen['Frente_Trabajo']).to_numpy()
    catalogo = mwd_clasificar(resumen)
    info = {'filas': int(n_filas), 'excluidas': int(n_excl), 'frentes': int(len(catalogo)),
            'barrenos': int(len(acum)), 'segundos': round(time.time() - t0, 1),
            'cortes': [float(v) for v in catalogo['rop_medio'].quantile([0.2, 0.4, 0.6, 0.8])]}
    log(f"MWD: catálogo listo, {info['frentes']} frentes, {info['barrenos']:,} barrenos, "
        f"{info['filas']:,} filas ({info['excluidas']:,} excluidas) en {info['segundos']} s")
    return catalogo, info


def mwd_cuantil_actividad(dist_params, distribucion, p):
    """Duracion (h) de una actividad en el percentil p de SU distribucion, con la misma semantica
    del motor: kwargs de scipy iguales, truncamiento de rango si existe y clamp a 0. Devuelve
    None si no se puede evaluar (el motor usa entonces el muestreo normal)."""
    if distribucion == 'Cte' or isinstance(dist_params, (int, float)):
        return float(dist_params) if isinstance(dist_params, (int, float)) else 1.0
    if distribucion not in DIST_OBJ_SCIPY:
        return None
    try:
        kwargs = _kwargs_scipy_desde_params(dist_params, distribucion)
        dist_obj = DIST_OBJ_SCIPY[distribucion]
        lo = dist_params.get('_trunc_lo') if isinstance(dist_params, dict) else None
        hi = dist_params.get('_trunc_hi') if isinstance(dist_params, dict) else None
        u = float(p)
        if lo is not None and hi is not None:
            c_lo, c_hi = dist_obj.cdf(lo, **kwargs), dist_obj.cdf(hi, **kwargs)
            if not (np.isfinite(c_lo) and np.isfinite(c_hi) and c_hi > c_lo):
                return None
            u = float(c_lo + p * (c_hi - c_lo))
        valor = float(dist_obj.ppf(u, **kwargs))
        if not np.isfinite(valor):
            return None
        if lo is not None and hi is not None:
            valor = min(max(valor, lo), hi)
        return max(0.0, valor)
    except Exception:
        return None


MODO_GEOLOGIA_3D = "Geología (color por calidad de roca)"
MWD_MAX_SIMS_LOG = 200   # se registran en el log los sorteos de las primeras N simulaciones (memoria)


def mwd_segmentos_fijos(tramos, largo):
    """Tramos (desde, hasta, clase) que cubren TODO el túnel. Huecos -> clase del tramo siguiente/anterior;
    'hasta' None = hasta el final del túnel. El final sin tramo toma la clase del último tramo."""
    tr = sorted(([max(float(d), 0.0), largo if h is None else min(float(h), largo), c] for d, h, c in tramos),
                key=lambda x: x[0])
    segs, cur = [], 0.0
    for d, h, c in tr:
        if d >= largo - 1e-9:
            continue
        if d > cur + 1e-9:
            segs.append([cur, d, segs[-1][2] if segs else c])
            cur = d
        d = max(d, cur)
        if h > d + 1e-9:
            segs.append([d, h, c])
            cur = h
    if segs and cur < largo - 1e-9:
        segs.append([cur, largo, segs[-1][2]])
    return [tuple(s) for s in segs]


def mwd_segmentos_aleatorios(largo, largo_tramo, n_cat, rng):
    """Tramos consecutivos de 'largo_tramo' m; cada uno con una referencia MWD al azar de toda la base."""
    segs, d = [], 0.0
    while d < largo - 1e-9:
        h = min(d + float(largo_tramo), largo)
        segs.append((d, h, int(rng.integers(n_cat))))
        d = h
    return segs


def mwd_preparar(cfg, frentes_nombres):
    """Valida la configuracion y precalcula estructuras. None si ningun tunel simulado tiene MWD."""
    log = []
    cfg['_log_geologia'] = log
    cat = cfg['catalogo'].reset_index(drop=True)
    idx_por_clase = {c: np.flatnonzero((cat['calidad_roca'] == c).to_numpy()) for c in MWD_CLASES}
    asignaciones = {}
    for f in frentes_nombres:
        a = cfg['asignaciones'].get(str(f))
        if not a or a.get('modo') == 'Sin MWD':
            continue
        if a['modo'] == 'Fija':
            for _, _, c in a['tramos']:
                if len(idx_por_clase.get(c, [])) == 0:
                    raise ValueError(f"La clase '{c}' del túnel '{f}' no tiene frentes en el catálogo MWD.")
        elif a['modo'] == 'Aleatoria' and float(a.get('largo_tramo', 0)) <= 0:
            raise ValueError(f"El túnel '{f}' tiene un largo de tramo aleatorio inválido.")
        asignaciones[str(f)] = a
    if not asignaciones:
        return None
    return {'cat_clase': cat['calidad_roca'].to_numpy(), 'cat_frente': cat['Frente_Trabajo'].astype(str).to_numpy(),
            'cat_rop': cat['rop_medio'].to_numpy(float), 'cat_p': cat['p_rop'].to_numpy(float),
            'idx_por_clase': idx_por_clase, 'asignaciones': asignaciones,
            'reglas': {k: v for k, v in cfg.get('reglas', {}).items() if v in ('directa', 'inversa')},
            'semiancho': float(cfg.get('semiancho', 0.05)), 'log': log, 'sims_log': MWD_MAX_SIMS_LOG}


class MWDEstadoFrente:
    """Perfil de roca de UN tunel en UNA replica: lista de tramos (desde, hasta, referencia MWD).
    Fija: tramos definidos por el usuario (una referencia de la clase por tramo).
    Aleatoria: tramos del largo elegido, cada uno con una referencia al azar de toda la base; se sortea una sola vez
    y queda igual en todas las simulaciones."""

    def __init__(self, prep, frente, rng, sim, largo):
        p, a = prep, prep['asignaciones'][frente]
        if a['modo'] == 'Fija':
            crudo = [(d, h, int(rng.choice(p['idx_por_clase'][c])), c) for d, h, c in mwd_segmentos_fijos(a['tramos'], largo)]
        else:
            # Aleatoria: se sortea UNA sola vez (primera simulación) y esa misma geología se repite en todas.
            _fijo = p.setdefault('perfil_aleatorio_fijo', {})
            if frente not in _fijo:
                _fijo[frente] = [(d, h, i, 'Aleatoria')
                                 for d, h, i in mwd_segmentos_aleatorios(largo, a['largo_tramo'], len(p['cat_clase']), rng)]
            crudo = _fijo[frente]
        self.segs = []
        for k, (d, h, i, etiqueta) in enumerate(crudo, 1):
            est = {'bloque': k, 'calidad': str(p['cat_clase'][i]), 'frente_mwd': str(p['cat_frente'][i]),
                   'rop_medio': float(p['cat_rop'][i]), 'p_rop': float(p['cat_p'][i])}
            self.segs.append((d, h, est))
            if sim < p['sims_log']:
                p['log'].append({'Simulacion': sim + 1, 'Tunel': frente, 'Bloque': k, 'Modo': a['modo'],
                                 'Clase_Asignada': etiqueta, 'Calidad_Roca': est['calidad'],
                                 'Frente_MWD_Ref': est['frente_mwd'], 'ROP_Medio': est['rop_medio'],
                                 'p_rop': est['p_rop'], 'Tramo_Desde_m': d, 'Tramo_Hasta_m': h, 'Largo_m': h - d})

    def cortes(self, metros, largo):
        """[(desde_m, hasta_m, estado)] de los tramos de roca que atraviesa el ciclo [metros, metros+largo]."""
        a, b = float(metros), float(metros) + max(float(largo or 0.0), 1e-6)
        out = [(max(d, a), min(h, b), e) for d, h, e in self.segs if min(h, b) - max(d, a) > 1e-12]
        return out or [(a, b, self.segs[-1][2])]

    def para_ciclo(self, metros, largo):
        """[(fraccion del ciclo, estado)] segun los metros que el ciclo recorre en cada roca."""
        cs = self.cortes(metros, largo)
        tot = sum(h - d for d, h, _ in cs)
        return [((h - d) / tot, e) for d, h, e in cs]


def mwd_runtime_sim(prep, ss, frentes_info, sim):
    """Estado aleatorio de UNA replica (semillas independientes por tunel)."""
    nombres = list(frentes_info.keys())
    hijos = ss.spawn(1)[0].spawn(len(nombres) + 1)
    estados = {}
    for i, f in enumerate(nombres):
        if str(f) in prep['asignaciones']:
            e = MWDEstadoFrente(prep, str(f), np.random.default_rng(hijos[i]), sim, float(frentes_info[f]['distancia']))
            if e.segs:
                estados[str(f)] = e
    return {'estados': estados, 'reglas': prep['reglas'], 'semiancho': prep['semiancho'],
            'rng_t': np.random.default_rng(hijos[-1]), 'ref_cache': prep.setdefault('ref_cache', {})}


def mwd_muestrear(rt, sim, frente, act, ciclo, duracion_base, metros=0.0, largo=None):
    """Entra: duracion ya muestreada por el motor. Sale: (duracion, columnas de trazabilidad).
    Si el tunel tiene MWD y la actividad regla directa/inversa, la duracion es el promedio ponderado por
    METROS de los cuantiles de cada roca que atraviesa el ciclo (ej. 1,5 m de roca mala + 1,5 m de buena
    en un ciclo de 3 m -> duracion intermedia). 'metros' = avance acumulado al inicio del ciclo."""
    est = rt['estados'].get(str(frente))
    if est is None:
        return duracion_base, {}
    cortes = est.cortes(metros, largo)
    _tot = sum(h - d for d, h, _ in cortes)
    partes = [((h - d) / _tot, e) for d, h, e in cortes]
    dom = max(partes, key=lambda x: x[0])[1]
    cols = {'Calidad_Roca': dom['calidad'], 'Frente_MWD_Ref': dom['frente_mwd'], 'p_rop_MWD': round(dom['p_rop'], 4),
            'Bloque_MWD': dom['bloque'], 'Posicion_m_MWD': round(float(metros), 2),
            'Mezcla_Roca': " / ".join(f"{e['calidad']} {f * 100:.0f}%" for f, e in partes)}
    regla = rt['reglas'].get(act['Actividad'], 'libre')
    if regla == 'libre':
        return duracion_base, cols
    valor, p_med, c_med, trs = 0.0, 0.0, 0.0, []
    for (f, e), (_d0, _d1, _) in zip(partes, cortes):
        centro = e['p_rop'] if regla == 'directa' else 1.0 - e['p_rop']
        ancho = min(rt['semiancho'], centro - 1e-9, 1.0 - 1e-9 - centro)
        p = float(rt['rng_t'].uniform(centro - ancho, centro + ancho))
        v = mwd_cuantil_actividad(act['Tiempo'], act['Distribucion'], p)
        if v is None:
            return duracion_base, cols
        valor += f * v
        c_med += f * centro
        trs.append([round(_d0, 3), round(_d1, 3), e['calidad'], round(e['p_rop'], 4), round(p, 4), round(v, 4), round(f * v, 4), round(float(largo or 0.0), 3)])
        p_med += f * p
    import json as _json
    cols['Tramos_MWD'] = _json.dumps(trs)
    clave = (act['Actividad'], str(act['Tiempo']))
    ref = rt['ref_cache'].get(clave)
    if ref is None:
        tp = act['Tiempo']
        med = mwd_cuantil_actividad(tp, act['Distribucion'], 0.5)
        lo = tp.get('_trunc_lo') if isinstance(tp, dict) else None
        hi = tp.get('_trunc_hi') if isinstance(tp, dict) else None
        ref = (float('nan') if med is None else float(med),
               float('nan') if lo is None else float(lo), float('nan') if hi is None else float(hi))
        rt['ref_cache'][clave] = ref
    cols.update({'Regla_MWD': regla, 'p_tiempo_MWD': round(p_med, 4), 'p_centro_MWD': round(c_med, 4), 'Dur_Ref_Mediana_h': round(ref[0], 4),
                 'Trunc_Lo_h': ref[1], 'Trunc_Hi_h': ref[2]})
    return valor, cols



@st.cache_data(show_spinner=False)
def _mwd_catalogo_desde_ruta(ruta, mtime, size):
    """Cache por (ruta, fecha de modificacion, tamano): el CSV de +300 MB se procesa una sola vez."""
    return mwd_catalogo_desde_csv(ruta, log=lambda m: log_evento('INFO', m))


def _mwd_confirmar_carga(info, origen, tam_bytes=None):
    """Registra en el panel de logs cuantos datos MWD se cargaron y confirma si la carga fue exitosa.
    Criterio de exito: al menos 1 fila leida, 1 fila valida, 1 frente y 1 barreno. Avisa si hubo filas
    excluidas. Se llama tras cada 'Procesar MWD' (tambien cuando el resultado vino de la cache)."""
    try:
        filas = int(info.get('filas', 0))
        excl = int(info.get('excluidas', 0))
        validas = filas - excl
        frentes = int(info.get('frentes', 0))
        barrenos = int(info.get('barrenos', 0))
        seg = info.get('segundos', 0)
        tam = f" ({tam_bytes / 1e6:.1f} MB)" if tam_bytes else ''
        nombre = os.path.basename(str(origen))
        if filas > 0 and validas > 0 and frentes > 0 and barrenos > 0:
            pct = 100.0 * validas / filas
            log_evento('SUCCESS',
                       f"Carga MWD exitosa: {nombre}{tam}. Datos cargados: {filas:,} filas leídas, "
                       f"{validas:,} válidas ({pct:.2f}%), {excl:,} excluidas, {frentes:,} frentes y "
                       f"{barrenos:,} barrenos. Tiempo de lectura: {seg} s.")
            if excl > 0:
                log_evento('WARNING',
                           f"MWD: {excl:,} filas se excluyeron por no tener frente, barreno o ROP numérico válido.")
        else:
            log_evento('ERROR',
                       f"Carga MWD NO exitosa: {nombre}{tam}. Filas leídas: {filas:,}, válidas: {validas:,}, "
                       f"frentes: {frentes:,}, barrenos: {barrenos:,}. Revise el formato del archivo.")
    except Exception as e:
        log_evento('ERROR', f"No se pudo confirmar la carga MWD: {e}")


import copy as _copy


def _mwd_largos_tuneles(frentes, actividades_nivel):
    """{tunel: largo en metros} desde la hoja Frentes del nivel."""
    try:
        nivel = actividades_nivel['Nivel'].iloc[0]
        df = st.session_state.df_frentes
        df = df[(df['Nivel'] == nivel) & (df['Frentes'].astype(str).isin(list(frentes)))]
        return {str(r.Frentes): float(calcular_distancia(r.Xi, r.Yi, r.Xf, r.Yf)) for r in df.itertuples()}
    except Exception:
        return {}


def mwd_config_para_simulacion(frentes_sim, actividades_nivel=None):
    """Config MWD lista para el motor, o None si no esta activa / no aplica a los tuneles simulados.
    Reglas: actividades con jumbo = inversa (automatico) + ajustes del usuario."""
    cfg = st.session_state.get('mwd_config')
    cat = st.session_state.get('mwd_catalogo')
    if not cfg or not cfg.get('activo'):
        return None
    if cat is None or len(cat) == 0:
        log_evento('WARNING', "Geología MWD activa pero no hay catálogo MWD cargado: se ignora en esta simulación.")
        return None
    sim_set = {str(f) for f in frentes_sim}
    asign = {f: a for f, a in cfg['asignaciones'].items() if f in sim_set and a.get('modo') != 'Sin MWD'}
    if not asign:
        log_evento('WARNING', "Geología MWD activa pero ninguno de los túneles seleccionados tiene geología asignada.")
        return None
    reglas = mwd_reglas_jumbo(actividades_nivel) if (cfg.get('auto_jumbo', True) and actividades_nivel is not None) else {}
    reglas.update(cfg.get('reglas_override', {}))
    reglas = {k: v for k, v in reglas.items() if v in ('directa', 'inversa')}
    log_evento('INFO', f"Geología MWD activa: {len(asign)} túneles, {len(cat)} frentes de referencia, "
                       f"reglas {reglas or 'ninguna (solo etiqueta la roca)'}")
    return {'activo': True, 'catalogo': cat, 'asignaciones': asign, 'reglas': reglas,
            'semiancho': cfg['semiancho'], 'avances_bloque': None, 'exigir_cambio': False, 'semilla': None}


def mwd_perfiles_vista(asignaciones, largos, cat, semilla=0):
    """{tunel: [(desde, hasta, clase)]} para dibujar la geologia configurada. Aleatoria = una muestra de vista previa."""
    rng = np.random.default_rng(int(semilla))
    clases = cat['calidad_roca'].astype(str).to_numpy()
    out = {}
    for t, a in asignaciones.items():
        largo = largos.get(str(t))
        if not largo:
            continue
        if a['modo'] == 'Fija':
            out[str(t)] = mwd_segmentos_fijos(a['tramos'], largo)
        elif a['modo'] == 'Aleatoria':
            out[str(t)] = [(d, h, str(clases[i]))
                           for d, h, i in mwd_segmentos_aleatorios(largo, a['largo_tramo'], len(clases), rng)]
    return out


def mwd_perfiles_desde_log(log, sim):
    """{tunel: [(desde, hasta, clase)]} de una simulacion, desde el registro de geologia sorteada."""
    sub = log[log['Simulacion'] == sim]
    return {str(t): [(float(r.Tramo_Desde_m), float(r.Tramo_Hasta_m), str(r.Calidad_Roca))
                     for r in g.sort_values('Tramo_Desde_m').itertuples()]
            for t, g in sub.groupby('Tunel', sort=False)}


def _mwd_fig_perfiles(perfiles, titulo):
    """Franjas horizontales (una por tunel) coloreadas por calidad de roca a lo largo de los metros."""
    fig = go.Figure()
    vistos = set()
    for t, segs in perfiles.items():
        for d, h, c in segs:
            fig.add_trace(go.Bar(
                y=[t], x=[h - d], base=[d], orientation='h', marker_color=MWD_COLORES.get(c, '#9e9e9e'),
                name=c, legendgroup=c, showlegend=c not in vistos,
                hovertemplate=f"{t}<br>{c}: {d:.1f} → {h:.1f} m<extra></extra>"))
            vistos.add(c)
    fig.update_layout(barmode='overlay', title=titulo, xaxis_title="Metros desde la boca del túnel", yaxis_title="",
                      height=max(220, 34 * len(perfiles) + 120), margin=dict(l=10, r=10, t=50, b=10),
                      yaxis=dict(autorange='reversed'), legend=dict(orientation='h', y=1.12))
    return fig


def _mwd_txt_geologia(a):
    if a['modo'] == 'Fija':
        if a.get('sub') == 'unica':
            return f"Todo el túnel: {a['tramos'][0][2]}"
        return " | ".join(f"{d:g}–{'fin' if h is None else format(h, 'g')} m: {c}" for d, h, c in a['tramos'])
    if a['modo'] == 'Aleatoria':
        return f"Aleatoria: un frente MWD al azar cada {a['largo_tramo']:g} m (se sortea una vez y es la misma en todas las simulaciones)"
    return "Sin MWD"



_MWD_PREFIJOS_WIDGET = ("mwd_modo_", "mwd_sub_", "mwd_clase_", "mwd_lt_", "mwd_ed_tr_")


def _mwd_reset_widgets(tunel=None):
    """Borra el estado de los widgets de un túnel (o de todos) para que reaparezcan vacíos."""
    for k in list(st.session_state.keys()):
        if k.startswith(_MWD_PREFIJOS_WIDGET) and (tunel is None or k.endswith("_" + str(tunel))):
            st.session_state.pop(k, None)
    if tunel is None:
        st.session_state.pop('mwd_tr_base', None)
    else:
        st.session_state.get('mwd_tr_base', {}).pop(tunel, None)


def _mwd_validar(t, a, clases_ok):
    """(errores, asignacion normalizada o None) de la geologia de un tunel."""
    if a['modo'] == 'Fija':
        tr = sorted([list(x) for x in a['tramos']], key=lambda x: x[0])
        err = []
        if not tr:
            err.append("Defina al menos un tramo.")
        for d, h, c in tr:
            if h is not None and h <= d:
                err.append(f"El tramo {d:g}-{h:g} m tiene 'Hasta' menor o igual que 'Desde'.")
            if c not in clases_ok:
                err.append(f"La clase '{c}' no tiene frentes en el catálogo MWD.")
        for (d1, h1, _), (d2, h2, _) in zip(tr, tr[1:]):
            if h1 is None or d2 < h1 - 1e-9:
                err.append(f"Los tramos desde {d1:g} y {d2:g} m se traslapan (o un tramo sin 'Hasta' no es el último).")
        if err:
            return err, None
        return [], {'modo': 'Fija', 'sub': a.get('sub', 'tramos'), 'tramos': tr}
    if a['modo'] == 'Aleatoria':
        return [], {'modo': 'Aleatoria', 'largo_tramo': float(a['largo_tramo'])}
    return [], None


def _mwd_fig_3d(perfiles, titulo, altura=520):
    """Tuneles 3D (geometria real) coloreados por calidad de roca. perfiles = {tunel: [(desde, hasta, clase)]}."""
    nivel = None
    try:
        df = st.session_state.df_frentes
        nombres = [str(n) for n in perfiles]
        df = df[df['Frentes'].astype(str).isin(nombres)]
        base = _base_tuneles_desde_df(df)
        tuneles = _armar_tuneles_3d(base, None, _dimensiones_3d(base))
        for t in tuneles:
            t['geologia'] = perfiles.get(str(t['nombre']))
        fig = graficar_tuneles_3d(tuneles, titulo=titulo, modo_color=MODO_GEOLOGIA_3D,
                                  mostrar_pendiente=False, mostrar_lineas=False)
        fig.update_layout(height=altura)
        return fig
    except Exception as e:
        log_evento('WARNING', f"No se pudo dibujar el 3D de geología: {e}")
        return None


def _mwd_mostrar_config_guardada():
    cfg = st.session_state.get('mwd_config')
    if not cfg:
        return
    st.markdown("#### Geología guardada por túnel")
    st.caption(f"Estado: {'ACTIVA' if cfg['activo'] else 'desactivada'} | semiancho {cfg['semiancho']:.2f} | "
               f"reglas manuales que modifican las automáticas: "
               f"{', '.join(f'{k} ({v})' for k, v in cfg.get('reglas_override', {}).items()) or 'ninguna'}")
    filas = [{'Túnel': t, 'Modo': a['modo'], 'Detalle': _mwd_txt_geologia(a)} for t, a in cfg['asignaciones'].items()]
    if filas:
        mostrar_tabla(pd.DataFrame(filas), width='stretch', hide_index=True)
    if st.button("Eliminar toda la geología guardada", key="mwd_btn_limpiar", type="primary",
                 help="Borra la geología guardada de todos los túneles. El catálogo MWD cargado se conserva."):
        st.session_state.mwd_config = None
        _mwd_reset_widgets()
        reiniciar_resultados_simulacion("limpiaste la configuración de geología MWD (Restricciones Geológicas)")
        st.rerun()


def _mwd_ui_menu10(frentes_disponibles, actividades_nivel):
    """Subseccion del Menu 10: geologia desde MWD historico, adicional a los ponderadores manuales."""
    st.markdown("---")
    subtitulo_menu(
        "Geología desde MWD histórico (opción adicional a los ponderadores)",
        ayuda="El MWD histórico clasifica frentes por ROP medio en 5 clases de roca (Muy buena a Muy mala). "
              "Usted elige a qué túneles aplica y qué roca hay en cada uno: Fija (una clase para todo el túnel o "
              "por tramos de metros) o Aleatoria (la roca se sortea una sola vez y se repite igual en todas las simulaciones). La roca de cada metro "
              "del túnel fija el percentil de duración de las actividades con jumbo: si un ciclo cruza dos rocas, "
              "su duración es el promedio ponderado por los metros de cada una. Los ponderadores manuales se "
              "aplican después.")
    ayuda_inline(
        "PASOS: (1) Cargue y procese el CSV MWD. (2) Elija los túneles. (3) Para cada túnel elija Fija "
        "(todo el túnel con una clase, o tramos 'de tal metro a tal metro'), Aleatoria o Sin MWD; la pantalla "
        "cambia según el modo. (4) Las actividades con jumbo se afectan automáticamente (roca mala = perforación "
        "más rápida). (5) Cada túnel se guarda solo; luego simule. Verá la geología en 3D aquí, en Geometría (Menú 15), "
        "en el Avance 3D (Menú 17) y en la pestaña 'Geología MWD'.",
        etiqueta="Instrucciones: cómo usar la geología MWD")
    frentes = [str(f) for f in frentes_disponibles]
    auto = mwd_reglas_jumbo(actividades_nivel)
    try:
        actividades = list(dict.fromkeys(
            actividades_nivel.sort_values('Secuencia')['Actividad'].dropna().astype(str)))
    except Exception:
        actividades = sorted(actividades_nivel['Actividad'].dropna().astype(str).unique())
    recursos_act = {}
    try:
        for a, r in actividades_nivel[['Actividad', 'Recurso']].dropna().astype(str).itertuples(index=False):
            recursos_act.setdefault(a, [])
            if r not in recursos_act[a]:
                recursos_act[a].append(r)
    except Exception:
        pass

    # ---------------------------- 1) Carga del MWD ----------------------------
    modo_fuente = st.radio("Origen del archivo MWD",
                           ["Subir archivo CSV"] if ES_NUBE else ["Ruta local del CSV (recomendado)", "Subir archivo CSV"],
                           key="mwd_fuente_modo", horizontal=True,
                           help="El CSV histórico pesa más de 300 MB. Con la ruta local se lee directo del disco, por "
                                "bloques y sin pérdida de información. Subir archivo requiere subir el límite de "
                                "carga de Streamlit (server.maxUploadSize en .streamlit/config.toml, por defecto 200 MB).")
    if ES_NUBE:
        st.info("Streamlit Cloud: el servidor no ve las carpetas de su equipo, por eso el MWD se carga subiendo el CSV. "
                "El límite de subida se define en .streamlit/config.toml del repositorio (maxUploadSize = 600).")
    if modo_fuente.startswith("Ruta"):
        ruta = st.text_input("Ruta completa del CSV MWD", value=st.session_state.get('mwd_ruta', ''),
                             key="mwd_ruta_input",
                             help="Ruta absoluta del archivo, con nombre y extensión. Ejemplo: "
                                  "/Users/usuario/Desktop/TESIS/BaseDatos_MWD_Final.csv. Debe tener las columnas "
                                  "Frente_Trabajo, ID_Barreno y ROP_Normalizado_Final.")
        if st.button("Procesar MWD", key="mwd_btn_ruta",
                     help="Lee el CSV por bloques, promedia el ROP por barreno y por frente y clasifica los frentes "
                          "en 5 clases. Se guarda en caché: si el archivo no cambia, no se vuelve a leer."):
            ruta_l = ruta.strip().strip('"').strip("'")
            if not os.path.isfile(ruta_l):
                st.error(f"No existe el archivo: {ruta_l}")
            else:
                try:
                    log_evento('INFO', f"Leyendo MWD ({os.path.getsize(ruta_l) / 1e6:.1f} MB): {ruta_l}")
                    with st.spinner("Leyendo y clasificando el MWD (puede tardar en archivos grandes)..."):
                        cat, info = _mwd_catalogo_desde_ruta(ruta_l, os.path.getmtime(ruta_l), os.path.getsize(ruta_l))
                    st.session_state.mwd_ruta = ruta_l
                    st.session_state.mwd_catalogo = cat
                    st.session_state.mwd_info = dict(info, origen=ruta_l)
                    st.session_state.mwd_info.update(mwd_vista_previa(ruta_l))
                    _mwd_confirmar_carga(info, ruta_l, os.path.getsize(ruta_l))
                    reiniciar_resultados_simulacion("cargaste un nuevo catálogo MWD (Restricciones Geológicas)")
                except Exception as e:
                    log_evento('ERROR', f"No se pudo procesar el MWD: {e}")
                    st.error(f"No se pudo procesar el MWD: {e}")
    else:
        archivo = st.file_uploader("Subir CSV MWD", type=['csv'], key="mwd_uploader",
                                   help="Archivo CSV del MWD histórico. Si pesa más de 200 MB use la ruta local.")
        if archivo is not None and st.button("Procesar MWD", key="mwd_btn_subida",
                                             help="Lee el CSV subido por bloques y clasifica los frentes en 5 clases."):
            try:
                log_evento('INFO', f"Leyendo MWD subido ({archivo.size / 1e6:.1f} MB): {archivo.name}")
                with st.spinner("Leyendo y clasificando el MWD..."):
                    cat, info = mwd_catalogo_desde_csv(archivo, log=lambda m: log_evento('INFO', m))
                st.session_state.mwd_catalogo = cat
                st.session_state.mwd_info = dict(info, origen=archivo.name)
                st.session_state.mwd_info.update(mwd_vista_previa(archivo))
                _mwd_confirmar_carga(info, archivo.name, archivo.size)
                reiniciar_resultados_simulacion("cargaste un nuevo catálogo MWD (Restricciones Geológicas)")
            except Exception as e:
                log_evento('ERROR', f"No se pudo procesar el MWD: {e}")
                st.error(f"No se pudo procesar el MWD: {e}")

    cat = st.session_state.get('mwd_catalogo')
    if cat is None or len(cat) == 0:
        st.info("Cargue y procese un archivo MWD para configurar la geología por túnel.")
        return
    info = st.session_state.get('mwd_info', {})
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Frentes MWD", f"{len(cat):,}", help="Cantidad de frentes históricos del MWD, cada uno con un ROP medio.")
    m2.metric("Barrenos", f"{int(cat['n_barrenos'].sum()):,}", help="Barrenos únicos usados. Cada barreno pesa igual dentro de su frente.")
    m3.metric("Filas leídas", f"{info.get('filas', 0):,}", help="Filas del CSV procesadas.")
    m4.metric("Filas excluidas", f"{info.get('excluidas', 0):,}", help="Filas sin identificador o sin ROP numérico finito.")
    st.caption("Las clases son RELATIVAS a esta base: P20/P40/P60/P80 del ROP medio por frente (igual peso a cada "
               "barreno y a cada frente). ROP <= P20 es Muy buena y ROP > P80 es Muy mala. Los empates reciben la "
               "misma clase, por lo que alguna clase puede quedar con pocos frentes o vacía.")
    resumen_clases = (cat.groupby('calidad_roca').agg(Frentes=('Frente_Trabajo', 'count'), ROP_min=('rop_medio', 'min'),
                                                      ROP_medio=('rop_medio', 'mean'), ROP_max=('rop_medio', 'max'))
                      .reindex(MWD_CLASES).reset_index().rename(columns={'calidad_roca': 'Clase'}))
    mostrar_tabla(resumen_clases.round(3), width='stretch', hide_index=True)
    _mwd_ui_explicacion(cat, info)
    with st.expander("Ver distribución y catálogo de frentes MWD"):
        tab_d, tab_b, tab_c = st.tabs(["Distribución", "Por clase", "Catálogo"])
        with tab_d:
            rop_v = cat['rop_medio'].to_numpy(float)
            paso = max((rop_v.max() - rop_v.min()) / 40.0, 1e-9)
            fig_h = go.Figure()
            for c in MWD_CLASES:
                sub_c = cat[cat['calidad_roca'] == c]
                if len(sub_c):
                    fig_h.add_trace(go.Histogram(x=sub_c['rop_medio'], name=c, marker_color=MWD_COLORES[c],
                                                 xbins=dict(start=float(rop_v.min()), end=float(rop_v.max()) + paso, size=paso)))
            for nombre_p, q in zip(["P20", "P40", "P60", "P80"], [0.2, 0.4, 0.6, 0.8]):
                xq = float(np.quantile(rop_v, q))
                fig_h.add_vline(x=xq, line_dash="dash", line_color="#555", annotation_text=nombre_p, annotation_position="top")
            fig_h.update_layout(barmode='stack', bargap=0.04, height=380, margin=dict(l=10, r=10, t=30, b=10),
                                xaxis_title="ROP medio por frente", yaxis_title="Cantidad de frentes",
                                legend=dict(orientation='h', y=-0.25, title_text=''))
            st.plotly_chart(fig_h, width='stretch', key="mwd_hist_catalogo")
            st.caption("Las líneas punteadas son los cortes P20, P40, P60 y P80 que definen las 5 clases. ROP bajo = roca muy buena.")
        with tab_b:
            st.plotly_chart(px.box(cat, x='calidad_roca', y='rop_medio', color='calidad_roca', color_discrete_map=MWD_COLORES,
                                   category_orders={'calidad_roca': MWD_CLASES}, points='all',
                                   labels={'calidad_roca': 'Clase', 'rop_medio': 'ROP medio por frente'}),
                            width='stretch', key="mwd_box_catalogo")
        with tab_c:
            mostrar_tabla(cat.sort_values('rop_medio').round(4), width='stretch', hide_index=True)

    # ---------------------------- 2) Geologia por tunel (cada tunel se guarda solo) ----------------------------
    largos = _mwd_largos_tuneles(frentes, actividades_nivel)
    clases_ok = [c for c in MWD_CLASES if (cat['calidad_roca'] == c).any()]
    if not st.session_state.get('mwd_config'):
        st.session_state.mwd_config = {'activo': True, 'asignaciones': {}, 'auto_jumbo': True,
                                       'reglas_override': {}, 'semiancho': 0.05}
    cfg = st.session_state.mwd_config
    cfg.setdefault('asignaciones', {})
    guard = cfg['asignaciones']
    if 'mwd_tr_base' not in st.session_state:
        st.session_state.mwd_tr_base = {}
    base_tr = st.session_state.mwd_tr_base

    if auto:
        detalle = ", ".join(f"{a} ({', '.join(recursos_act.get(a, []))})" for a in auto)
        st.info(f"Actividades con jumbo detectadas automáticamente (regla inversa: más ROP, menos horas): {detalle}.")
    else:
        st.warning("No se detectaron actividades con recurso jumbo en este nivel. Defina reglas en 'Opciones generales'.")

    st.markdown("**Defina la geología túnel por túnel. Cada cambio válido se guarda automáticamente.**")

    def _etq(t):
        a = guard.get(t)
        return f"{t}  [guardado: {a['modo']}]" if a else f"{t}  [sin geología]"

    t_cfg = st.selectbox("Túnel a configurar", frentes, key="mwd_tunel_cfg", format_func=_etq,
                         help="Elija un túnel, defina su geología y pase al siguiente. Lo que queda guardado se indica entre corchetes.")
    largo_t = largos.get(str(t_cfg))
    d_prev = guard.get(t_cfg) or {'modo': 'Sin MWD'}
    st.caption(f"Largo del túnel {t_cfg}: {largo_t:.1f} m" if largo_t else f"No se pudo calcular el largo de {t_cfg}.")
    modo = st.radio("Tipo de geología de este túnel", MWD_MODOS, horizontal=True, key=f"mwd_modo_{t_cfg}",
                    index=MWD_MODOS.index(d_prev.get('modo', 'Sin MWD')),
                    help="Fija: usted define qué roca hay (todo el túnel o por tramos de metros). Aleatoria: la roca se "
                         "sortea una sola vez de toda la base MWD y queda igual en todas las simulaciones. Sin MWD: el túnel no usa esta opción.")
    nuevo = {'modo': 'Sin MWD'}

    if modo == 'Fija':
        sub = st.radio("¿Cómo define la roca?", ["Todo el túnel con una sola clase", "Por tramos (metros)"],
                       horizontal=True, key=f"mwd_sub_{t_cfg}",
                       index=0 if d_prev.get('sub', 'unica') == 'unica' else 1)
        if sub.startswith("Todo"):
            c0 = d_prev['tramos'][0][2] if d_prev.get('sub') == 'unica' and d_prev.get('tramos') else clases_ok[len(clases_ok) // 2]
            clase = st.selectbox("Clase de roca de todo el túnel", clases_ok, key=f"mwd_clase_{t_cfg}",
                                 index=clases_ok.index(c0) if c0 in clases_ok else 0)
            nuevo = {'modo': 'Fija', 'sub': 'unica', 'tramos': [[0.0, None, clase]]}
        else:
            if t_cfg not in base_tr:
                if d_prev.get('modo') == 'Fija' and d_prev.get('sub') == 'tramos' and d_prev.get('tramos'):
                    filas0 = d_prev['tramos']
                else:
                    filas0 = [[0.0, largo_t, clases_ok[len(clases_ok) // 2]]]
                base_tr[t_cfg] = pd.DataFrame({
                    'Desde_m': [float(f[0]) for f in filas0],
                    'Hasta_m': [np.nan if f[1] is None else float(f[1]) for f in filas0],
                    'Clase': [str(f[2]) for f in filas0]})
            st.caption("Una fila por tramo: de qué metro a qué metro hay cada clase de roca. 'Hasta' vacío = hasta el final "
                       "del túnel. Lo que no cubran los tramos toma la clase del tramo vecino. Para agregar un tramo use "
                       "la fila vacía al final de la tabla.")
            ed = st.data_editor(
                base_tr[t_cfg], num_rows="dynamic", hide_index=True, width='stretch', key=f"mwd_ed_tr_{t_cfg}",
                column_config={
                    'Desde_m': st.column_config.NumberColumn('Desde (m)', min_value=0.0, step=0.5, format="%.2f"),
                    'Hasta_m': st.column_config.NumberColumn('Hasta (m)', min_value=0.0, step=0.5, format="%.2f"),
                    'Clase': st.column_config.SelectboxColumn('Clase de roca', options=clases_ok, required=True)})
            tramos = []
            for r in ed.itertuples(index=False):
                if pd.isna(r.Desde_m) or pd.isna(r.Clase):
                    continue
                tramos.append([float(r.Desde_m), None if pd.isna(r.Hasta_m) else float(r.Hasta_m), str(r.Clase)])
            nuevo = {'modo': 'Fija', 'sub': 'tramos', 'tramos': tramos}
    elif modo == 'Aleatoria':
        lt = st.number_input("Largo de cada tramo aleatorio (m)", min_value=0.5, step=0.5, format="%.1f",
                             value=float(d_prev.get('largo_tramo', 10.0)), key=f"mwd_lt_{t_cfg}",
                             help="Cada simulación divide el túnel en tramos de este largo y asigna a cada uno la roca de un "
                                  "frente MWD sorteado de toda la base. Un tramo corto cambia más seguido de roca.")
        n_tr = int(np.ceil(largo_t / lt)) if largo_t else 0
        st.info(f"En cada simulación se sortean {n_tr or '—'} tramos de {lt:g} m con roca de toda la base MWD. "
                "El 3D de abajo muestra un sorteo de ejemplo; después de simular, la pestaña 'Geología MWD' y el "
                "Avance 3D (Menú 17) muestran cómo quedó cada sorteo.")
        if st.button("Volver a sortear el ejemplo", key=f"mwd_resort_{t_cfg}"):
            st.session_state.mwd_vista_sem = st.session_state.get('mwd_vista_sem', 0) + 1
        nuevo = {'modo': 'Aleatoria', 'largo_tramo': float(lt)}
    else:
        st.info("Este túnel no usa geología MWD: se simula con sus distribuciones normales de tiempo.")

    # ---- guardado automatico ----
    errores, norm = _mwd_validar(t_cfg, nuevo, clases_ok)
    if errores:
        st.error("No guardado: " + " ".join(errores))
    else:
        if norm != guard.get(t_cfg):
            if norm is None:
                guard.pop(t_cfg, None)
            else:
                guard[t_cfg] = norm
            log_evento('INFO', f"Geología MWD de {t_cfg} guardada: {nuevo['modo']}")
            reiniciar_resultados_simulacion(f"cambiaste la geología MWD del túnel {t_cfg} (Restricciones Geológicas)")
        st.caption("Guardado automáticamente." if norm else "Sin geología MWD en este túnel.")

    def _borrar_tunel(t):
        st.session_state.mwd_config['asignaciones'].pop(t, None)
        _mwd_reset_widgets(t)

    b1, b2 = st.columns(2)
    b1.button(f"Eliminar la geología guardada de {t_cfg}", key="mwd_btn_borrar_tunel", on_click=_borrar_tunel,
              args=(t_cfg,), disabled=t_cfg not in guard,
              help="Quita la geología guardada de este túnel y deja sus controles en blanco.")
    otros = [f for f in frentes if f != t_cfg]
    with b2.popover("Copiar esta geología a otros túneles", disabled=norm is None):
        destinos = st.multiselect("Túneles de destino", otros, key="mwd_copiar_dest")
        if st.button("Copiar", key="mwd_btn_copiar", disabled=not destinos):
            for t in destinos:
                guard[t] = _copy.deepcopy(norm)
                _mwd_reset_widgets(t)
            reiniciar_resultados_simulacion("copiaste geología MWD a otros túneles (Restricciones Geológicas)")
            st.rerun()

    # ---- vista 3D: geometria + geologia ----
    st.markdown("**Vista 3D del túnel (geometría y geología)**")
    ver_otros = st.checkbox("Mostrar también los demás túneles con geología guardada", value=False, key="mwd_3d_otros")
    asig_vista = {}
    if ver_otros:
        asig_vista.update({t: a for t, a in guard.items() if t != t_cfg})
    if norm is not None:
        asig_vista[t_cfg] = norm
    perf = mwd_perfiles_vista(asig_vista, largos, cat, st.session_state.get('mwd_vista_sem', 0)) if asig_vista else {}
    if perf:
        fig3 = _mwd_fig_3d(perf, "Geología MWD: rojo = muy mala, naranja = mala, amarillo = regular, verde claro = buena, verde = muy buena")
        if fig3 is not None:
            st.plotly_chart(fig3, width='stretch', key="mwd_3d_cfg")
        else:
            st.warning("No se pudo dibujar el 3D de este túnel. Revise la hoja de Frentes.")
    else:
        st.caption("Defina una geología Fija o Aleatoria para verla en 3D.")

    # ---- opciones generales (se guardan solas) ----
    with st.expander("Opciones generales: uso en la simulación, reglas por actividad y ancho del sorteo"):
        activo = st.checkbox("Usar geología MWD en la simulación", value=cfg.get('activo', True), key="mwd_activo",
                             help="Si se desmarca, la geología guardada se conserva pero no se aplica al simular.")
        semiancho = st.number_input("Semiancho (puntos percentiles)", min_value=0.0, max_value=0.49,
                                    value=float(cfg.get('semiancho', 0.05)), step=0.01, format="%.2f", key="mwd_semiancho",
                                    help="Variabilidad de la duración alrededor del percentil que fija la roca. 0,05 = ±5 puntos percentiles.")
        st.caption("Las actividades con recurso jumbo ya vienen en 'inversa' (más ROP, menos horas: roca mala se perfora más rápido). "
                   "Directa: más ROP, más horas. Libre: la actividad no se modifica.")
        over_prev = cfg.get('reglas_override', {})
        df_reglas = pd.DataFrame({
            'Actividad': actividades,
            'Recursos': [", ".join(recursos_act.get(a, [])) or "sin recurso" for a in actividades],
            'Regla': [over_prev.get(a, auto.get(a, 'libre')) for a in actividades]})
        reglas_ed = st.data_editor(
            df_reglas, hide_index=True, width='stretch', disabled=['Actividad', 'Recursos'], key="mwd_ed_reglas",
            column_config={'Regla': st.column_config.SelectboxColumn('Regla', options=MWD_REGLAS, required=True)})
        override = {str(r['Actividad']): r['Regla'] for _, r in reglas_ed.iterrows()
                    if r['Regla'] != auto.get(str(r['Actividad']), 'libre')}
        if (bool(activo) != cfg.get('activo', True) or abs(float(semiancho) - float(cfg.get('semiancho', 0.05))) > 1e-12
                or override != over_prev):
            cfg.update({'activo': bool(activo), 'semiancho': float(semiancho), 'reglas_override': override})
            reiniciar_resultados_simulacion("cambiaste las opciones generales de geología MWD (Restricciones Geológicas)")
    reglas_efec = {**auto, **cfg.get('reglas_override', {})}
    if guard and not any(v in ('directa', 'inversa') for v in reglas_efec.values()):
        st.warning("Ninguna actividad tiene regla directa/inversa: el MWD solo etiquetará la roca y no modificará tiempos.")

    _mwd_mostrar_config_guardada()


def _mwd_ui_perfil_resultados(log, sim=None):
    """Resultados: como quedo sorteada la geologia a lo largo de cada tunel en una simulacion (3D)."""
    if log is None or len(log) == 0 or 'Tramo_Desde_m' not in log.columns:
        return
    st.markdown("#### Geología a lo largo de cada túnel (una simulación)",
                help="Túneles en 3D coloreados por calidad de roca, de la boca al frente. Verde = muy buena, rojo = muy mala.")
    if sim is None:
        sims = sorted(int(s) for s in log['Simulacion'].unique())
        sim = st.selectbox("Simulación a revisar", sims, key="mwd_res_sim_perfil")
    perf = mwd_perfiles_desde_log(log, sim)
    fig3 = _mwd_fig_3d(perf, f"Geología sorteada: simulación {sim}")
    if fig3 is not None:
        st.plotly_chart(fig3, width='stretch', key="mwd_res_perfil")
    st.caption(f"Se registran las primeras {MWD_MAX_SIMS_LOG} simulaciones.")
    with st.expander("Detalle de los tramos de esta simulación (referencia MWD usada)"):
        mostrar_tabla(log[log['Simulacion'] == sim][['Tunel', 'Tramo_Desde_m', 'Tramo_Hasta_m', 'Calidad_Roca',
                                                     'Frente_MWD_Ref', 'ROP_Medio']].round(3),
                      width='stretch', hide_index=True)


@st.cache_data(show_spinner=False)
def _mwd_resumen_duraciones(_df, run_id, tunel=None, sim=None):
    """Efecto del MWD en las actividades con regla (jumbo): percentil usado, duracion obtenida y
    comparacion contra la mediana de la misma distribucion (dentro del rango restringido, si existe)."""
    if _df is None or 'Regla_MWD' not in _df.columns:
        return None
    if tunel is not None:
        _df = _df[_df['Resource_Destination_Tunnel'].astype(str) == str(tunel)]
    if sim is not None:
        _df = _df[_df['Simulation_ID'] == sim]
    base = ['Actividad', 'Regla_MWD', 'Calidad_Roca', 'Duration_Muestreada', 'p_rop_MWD', 'p_tiempo_MWD']
    extra = [c for c in ['Dur_Ref_Mediana_h', 'Trunc_Lo_h', 'Trunc_Hi_h', 'Posicion_m_MWD', 'Frente_MWD_Ref']
             if c in _df.columns]
    d = _df.loc[(_df['type'] == 'activity') & _df['Regla_MWD'].notna(), base + extra]
    if d.empty:
        return None
    g = d.groupby(['Actividad', 'Regla_MWD', 'Calidad_Roca'], observed=True)
    res = g.agg(Muestras=('Duration_Muestreada', 'count'), p_rop_medio=('p_rop_MWD', 'mean'),
                p_tiempo_medio=('p_tiempo_MWD', 'mean'), Media_h=('Duration_Muestreada', 'mean'),
                P10_h=('Duration_Muestreada', lambda s: s.quantile(0.10)),
                P50_h=('Duration_Muestreada', 'median'),
                P90_h=('Duration_Muestreada', lambda s: s.quantile(0.90))).reset_index()
    if 'Dur_Ref_Mediana_h' in d.columns:
        ref = g['Dur_Ref_Mediana_h'].mean().reset_index(name='Ref_Mediana_h')
        res = res.merge(ref, on=['Actividad', 'Regla_MWD', 'Calidad_Roca'])
        res['Variacion_vs_mediana_pct'] = np.where(res['Ref_Mediana_h'] > 0,
                                                   (res['Media_h'] / res['Ref_Mediana_h'] - 1.0) * 100.0, np.nan)
    if 'Trunc_Lo_h' in d.columns:
        rg = d.groupby('Actividad').agg(lo=('Trunc_Lo_h', 'min'), hi=('Trunc_Hi_h', 'max'))
        rango = {a: (f"{r.lo:.2f} a {r.hi:.2f} h" if pd.notna(r.lo) and pd.notna(r.hi) else "sin restricción")
                 for a, r in rg.iterrows()}
        res['Rango_restringido'] = res['Actividad'].map(rango)
    res['Calidad_Roca'] = pd.Categorical(res['Calidad_Roca'], categories=MWD_CLASES, ordered=True)
    res = res.sort_values(['Actividad', 'Calidad_Roca']).reset_index(drop=True)
    muestra = d.sample(min(len(d), 60000), random_state=0) if len(d) > 60000 else d
    return res, muestra


def _mwd_rgba(color, alfa):
    """Color (hex '#rrggbb' o 'rgb(r,g,b)') a 'rgba(r,g,b,a)'."""
    c = str(color).strip()
    if c.startswith('#') and len(c) >= 7:
        return f"rgba({int(c[1:3], 16)},{int(c[3:5], 16)},{int(c[5:7], 16)},{alfa})"
    nums = re.findall(r"[\d.]+", c)
    if len(nums) >= 3:
        return f"rgba({int(float(nums[0]))},{int(float(nums[1]))},{int(float(nums[2]))},{alfa})"
    return f"rgba(120,120,120,{alfa})"


def _mwd_leyenda_abajo(fig, margen=130):
    """Leyenda horizontal bajo el grafico, separada del titulo del eje X (evita que se encimen)."""
    fig.update_layout(legend=dict(orientation='h', yanchor='top', y=-0.32, x=0, title_text=''), margin=dict(b=margen))
    fig.update_xaxes(title_standoff=8)


def _mwd_graf_efecto(res_d, k):
    """Efecto en actividades con jumbo: variacion vs mediana, tabla y cajas por clase."""
    if res_d is None:
        st.caption("No hay actividades con regla directa/inversa en este alcance: el MWD no modificó duraciones.")
        return
    resumen, muestra = res_d
    st.markdown("#### Efecto en las actividades con jumbo: percentil aplicado y duración resultante",
                help="Por actividad y clase de roca: percentil de tiempo aplicado, duración obtenida y diferencia contra la "
                     "mediana de la misma distribución.")
    st.caption("Regla inversa: p_tiempo = 1 - p_rop. Roca muy mala (ROP alto) da duraciones bajas; roca muy buena da duraciones "
               "altas. La regla directa invierte la relación. Si la actividad tiene rango restringido, el percentil se aplica "
               "dentro de ese rango.")
    if 'Variacion_vs_mediana_pct' in resumen.columns:
        fig_v = px.bar(resumen, x='Calidad_Roca', y='Variacion_vs_mediana_pct', color='Actividad', barmode='group',
                       category_orders={'Calidad_Roca': MWD_CLASES},
                       labels={'Calidad_Roca': 'Clase de roca', 'Variacion_vs_mediana_pct': 'Variación de la duración media vs mediana (%)'})
        st.plotly_chart(fig_v, width='stretch', key=f"mwd_res_variacion_{k}")
    mostrar_tabla(resumen.round(3), width='stretch', hide_index=True)
    fig_b = px.box(muestra, x='Calidad_Roca', y='Duration_Muestreada', color='Actividad',
                   category_orders={'Calidad_Roca': MWD_CLASES},
                   labels={'Calidad_Roca': 'Clase de roca', 'Duration_Muestreada': 'Duración (h)'})
    st.plotly_chart(fig_b, width='stretch', key=f"mwd_res_box_{k}")
    if 'Posicion_m_MWD' in muestra.columns and muestra['Posicion_m_MWD'].notna().any():
        st.markdown("#### Duración según la posición en el túnel",
                    help="Cada punto es una ejecución de una actividad con jumbo; el eje horizontal es el avance acumulado "
                         "al iniciar el ciclo. Se ve cómo la duración cambia al cambiar la roca.")
        fig_p = px.scatter(muestra.sample(min(len(muestra), 8000), random_state=1), x='Posicion_m_MWD',
                           y='Duration_Muestreada', color='Calidad_Roca', symbol='Actividad',
                           color_discrete_map=MWD_COLORES, category_orders={'Calidad_Roca': MWD_CLASES},
                           labels={'Posicion_m_MWD': 'Avance acumulado (m)', 'Duration_Muestreada': 'Duración (h)'})
        st.plotly_chart(fig_p, width='stretch', key=f"mwd_res_pos_{k}")


def _mwd_pct_clases(log):
    cnt = (log.groupby(['Tunel', 'Calidad_Roca'])['Largo_m'].sum().unstack(fill_value=0)
           .reindex(columns=MWD_CLASES, fill_value=0))
    return cnt.div(cnt.sum(axis=1), axis=0) * 100


def _mwd_efecto_actividades(df_tr, usada, k, tunel=None, sim=None, ciclo=None):
    """Efecto del MWD en las actividades con regla: sorteo (p_rop -> ventana -> duracion), impacto por clase de roca
    y duracion segun la posicion en el tunel. Se elige el total de las actividades con regla o una actividad."""
    if df_tr is None or 'Regla_MWD' not in df_tr.columns or 'p_centro_MWD' not in df_tr.columns:
        return
    d = df_tr[(df_tr['type'] == 'activity') & df_tr['Regla_MWD'].notna()]
    if tunel is not None:
        d = d[d['Resource_Destination_Tunnel'].astype(str) == str(tunel)]
    if sim is not None:
        d = d[d['Simulation_ID'] == sim]
    if ciclo is not None:
        d = d[d['Ciclo_Frente'] == ciclo]
    if d.empty:
        return
    st.markdown("#### Efecto del MWD en las actividades con jumbo",
                help="Elija una actividad con regla MWD. Pestaña 1: cómo se sorteó. "
                     "Pestaña 2: impacto según la roca. Pestaña 3: cambio con la posición en el túnel.")
    acts = list(dict.fromkeys(d['Actividad'].astype(str)))
    # Se quitó la opción "Total (todas las actividades con regla MWD)": mezclaba actividades y confundía.
    opcion = st.selectbox("Actividad", acts, key=f"mwd_ver_act_{k}",
                          help="Elija la actividad asociada al jumbo cuyo efecto MWD quiere revisar.")
    total = False
    da = d if total else d[d['Actividad'].astype(str) == opcion]
    sa = float(usada.get('semiancho', 0.05))
    da = da.copy()
    ref_fila = da['Dur_Ref_Mediana_h'].astype(float) if 'Dur_Ref_Mediana_h' in da.columns else pd.Series(np.nan, index=da.index)
    da['dur_rel'] = np.where(ref_fila > 0, (da['Duration_Muestreada'] / ref_fila - 1.0) * 100.0, np.nan)
    ycol, ylab = ('dur_rel', 'Variación respecto a la mediana de entrada (%)') if total else ('Duration_Muestreada', 'Duración (h)')
    da['p_rop_eq'] = np.where(da['Regla_MWD'] == 'directa', da['p_centro_MWD'], 1.0 - da['p_centro_MWD'])
    regla = "mixta" if da['Regla_MWD'].nunique() > 1 else str(da['Regla_MWD'].iloc[0])
    dentro = (da['p_tiempo_MWD'] - da['p_centro_MWD']).abs() <= sa + 1e-3
    dur = da[ycol].dropna()
    en_rango = np.nan
    if 'Trunc_Lo_h' in da.columns:
        con = da['Trunc_Lo_h'].notna() & da['Trunc_Hi_h'].notna()
        if con.any():
            dd = da[con]
            en_rango = float(((dd['Duration_Muestreada'] >= dd['Trunc_Lo_h'] - 1e-6) &
                              (dd['Duration_Muestreada'] <= dd['Trunc_Hi_h'] + 1e-6)).mean() * 100)
    mostrar_tarjetas_metricas([
        ("Ejecuciones", f"{len(da):,}", "", False, "actividad por avance por simulación"),
        ("p sorteado dentro de la ventana", f"{dentro.mean() * 100:.1f}", "%", False, "debe ser 100%"),
        ("Simulaciones" if sim is None else "Simulación", f"{da['Simulation_ID'].nunique():,}" if sim is None else f"{sim}", "", False),
        ("Dentro del rango restringido", "sin restricción" if np.isnan(en_rango) else f"{en_rango:.1f}", "" if np.isnan(en_rango) else "%", False),
    ], ancho_max=250)
    u = "%" if total else " h"
    fmt_ = (lambda v: f"{v:+.1f}") if total else (lambda v: f"{v:.2f}")
    que = ("variación de la duración respecto a la mediana de entrada (0% = igual a la mediana, negativo = más rápido, positivo = más lento)"
           if total else "duración en horas")
    if sim is None and len(dur):
        st.caption(f"Estadísticos de la {que}, de una ejecución (una actividad en un avance de un túnel en una simulación), sobre todas las simulaciones.")
        mostrar_tarjetas_metricas([
            ("Esperanza", fmt_(dur.mean()), u, True), ("P10", fmt_(dur.quantile(.1)), u, False),
            ("P50 (mediana)", fmt_(dur.quantile(.5)), u, True), ("P90", fmt_(dur.quantile(.9)), u, False),
            ("Mínimo", fmt_(dur.min()), u, False), ("Máximo", fmt_(dur.max()), u, False)], ancho_max=170)
    elif len(dur):
        st.caption(f"Valores directos de esta simulación (no son percentiles): {que}.")
        mostrar_tarjetas_metricas([
            ("Mínimo", fmt_(dur.min()), u, False), ("Media", fmt_(dur.mean()), u, True), ("Máximo", fmt_(dur.max()), u, False)], ancho_max=200)
    # El gráfico por metro es de UN túnel: en el resumen global (varios túneles mezclados) no se muestra.
    hay_metro = tunel is not None
    _tabs = st.tabs(["1. Verificación del sorteo", "2. Ventana en la distribución", "3. Impacto por clase de roca"]
                    + (["4. Por metro de avance"] if hay_metro else []))
    tab1, tabv, tab2 = _tabs[0], _tabs[1], _tabs[2]
    tab3 = _tabs[3] if hay_metro else None
    muestra = da.sample(min(len(da), 4000), random_state=3) if len(da) > 4000 else da
    with tab1:
        pres = [c_ for c_ in MWD_CLASES if (da['Calidad_Roca'] == c_).any()]
        f1, f2 = st.columns([3, 2])
        sel1 = f1.multiselect("Rocas a mostrar", pres, default=pres, key=f"mwd_f1r_{k}_{opcion}")
        ver_vent = f2.checkbox("Mostrar las líneas de la ventana (± semiancho)", value=True, key=f"mwd_f1v_{k}_{opcion}")
        # Un punto por ROCA atravesada (no por ciclo): cada punto cae en SU roca, con el p_rop, el p sorteado
        # y la duración de esa roca (se lee de Tramos_MWD). Así color, posición y curva siempre coinciden.
        ms = muestra
        if 'Tramos_MWD' in muestra.columns:
            _filas = []
            _refs = muestra['Dur_Ref_Mediana_h'] if 'Dur_Ref_Mediana_h' in muestra.columns else pd.Series(np.nan, index=muestra.index)
            for _tr, _reg, _ref in zip(muestra['Tramos_MWD'], muestra['Regla_MWD'], _refs):
                if not isinstance(_tr, str):
                    continue
                try:
                    _lst = json.loads(_tr)
                except Exception:
                    continue
                for _d0, _d1, _cal, _pr, _pt, _v, *_resto in _lst:
                    _ctr = _pr if _reg == 'directa' else 1.0 - _pr
                    _rel = (_v / _ref - 1.0) * 100.0 if (_ref == _ref and _ref > 0) else np.nan
                    _filas.append((_cal, _pr, _ctr, _pt, _v, _rel))
            if _filas:
                ms = pd.DataFrame(_filas, columns=['Calidad_Roca', 'p_rop_eq', 'p_centro_MWD', 'p_tiempo_MWD',
                                                   'Duration_Muestreada', 'dur_rel'])
                if len(ms) > 6000:
                    ms = ms.sample(6000, random_state=3)
        from plotly.subplots import make_subplots
        fila = None
        if not total:
            df_act = st.session_state.get('df_actividades_simuladas', pd.DataFrame())
            if df_act is not None and not df_act.empty and 'Actividad' in df_act.columns:
                m = df_act[df_act['Actividad'].astype(str) == opcion]
                fila = None if m.empty else m.iloc[0]
        fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1, subplot_titles=(
            "Centro de la ventana vs percentil de tiempo sorteado" if total else "p_rop de la roca vs percentil de tiempo sorteado",
            "Distribución acumulada de la variación de la duración, por roca" if total else f"Duraciones obtenidas en la distribución de {opcion}"))
        xs = np.linspace(0.0, 1.0, 101)
        if total or regla == 'directa':
            centro = xs
        else:
            centro = 1.0 - xs
        for off in ((sa, -sa) if ver_vent else ()):
            fig.add_trace(go.Scatter(x=xs, y=np.clip(centro + off, 0, 1), mode='lines', showlegend=False,
                                     line=dict(color='#555', dash='dash', width=1)), row=1, col=1)
        xcol = 'p_centro_MWD' if total else 'p_rop_eq'
        for c in [c_ for c_ in MWD_CLASES if c_ in sel1]:
            mc = ms[ms['Calidad_Roca'] == c]
            if len(mc):
                fig.add_trace(go.Scatter(x=mc[xcol], y=mc['p_tiempo_MWD'], mode='markers', name=c, legendgroup=c,
                                         marker=dict(color=MWD_COLORES[c], size=6, opacity=0.7)), row=1, col=1)
        pdf_ok = False
        if fila is not None:
            try:
                x_vals = np.linspace(0.0, max(float(ms['Duration_Muestreada'].max()) * 1.35, 1e-3), 400)
                y_pdf = calcular_pdf_teorica(fila['Tiempo'], fila['Distribucion'], x_vals)
                if y_pdf is not None:
                    fig.add_trace(go.Scatter(x=x_vals, y=y_pdf, mode='lines', name=f"Distribución de entrada ({fila['Distribucion']})",
                                             line=dict(color='#F2A33A', width=2), fill='tozeroy',
                                             fillcolor='rgba(242,163,58,0.25)'), row=1, col=2)
                    pdf_ok = True
                    for c in [c_ for c_ in MWD_CLASES if c_ in sel1]:
                        mc = ms[ms['Calidad_Roca'] == c]
                        if len(mc):
                            yy = calcular_pdf_teorica(fila['Tiempo'], fila['Distribucion'], mc['Duration_Muestreada'].to_numpy(float))
                            fig.add_trace(go.Scatter(x=mc['Duration_Muestreada'], y=yy, mode='markers', name=c, legendgroup=c,
                                                     showlegend=False, marker=dict(color=MWD_COLORES[c], size=6, opacity=0.7,
                                                                                   line=dict(color='white', width=0.5))), row=1, col=2)
            except Exception:
                pdf_ok = False
        if not pdf_ok:
            for c in [c_ for c_ in MWD_CLASES if c_ in sel1]:
                vals = np.sort(ms.loc[ms['Calidad_Roca'] == c, ycol].dropna().to_numpy(float))
                if len(vals):
                    fig.add_trace(go.Scatter(x=vals, y=np.arange(1, len(vals) + 1) / len(vals) * 100, mode='lines', name=c,
                                             legendgroup=c, showlegend=False, line=dict(color=MWD_COLORES[c], width=2.5, shape='hv'),
                                             hovertemplate=f"{c}<br>Duración: %{{x:.2f}}<br>Acumulado: %{{y:.0f}}%<extra></extra>"),
                                  row=1, col=2)
            if total:
                fig.add_vline(x=0.0, line_dash="dash", line_color="#555", row=1, col=2)
        fig.update_xaxes(title_text="Centro de la ventana (p de tiempo)" if total else "p_rop equivalente de la roca sorteada", range=[0, 1], row=1, col=1)
        fig.update_yaxes(title_text="Percentil de tiempo sorteado", range=[0, 1], row=1, col=1)
        fig.update_xaxes(title_text=ylab, row=1, col=2)
        fig.update_yaxes(title_text="Densidad de probabilidad" if pdf_ok else "% acumulado de ejecuciones", row=1, col=2)
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=60, b=10), legend=dict(orientation='h', y=-0.2, title_text=''))
        _mwd_leyenda_abajo(fig)
        st.plotly_chart(fig, width='stretch', key=f"mwd_ver_fig_{k}_{opcion}")
        if total:
            st.caption(f"Semiancho {sa:.2f}. Con varias actividades (reglas directa e inversa) el eje X es el centro de la ventana de tiempo: "
                       "los puntos deben quedar entre las dos diagonales. A la derecha, 0% es la mediana de entrada de cada actividad (negativo = más rápido que la mediana). "
                       "Para ver el p_rop y la curva de una actividad, elíjala arriba.")
        else:
            st.caption(f"Regla {regla}: el centro de la ventana es " + ("p_rop." if regla == 'directa' else "1 - p_rop.") +
                       f" Semiancho {sa:.2f}. Cada punto es una roca atravesada por un ciclo (si el ciclo cruza dos rocas, aparecen dos puntos, cada uno en su roca). Deben quedar entre las dos líneas punteadas y sobre la curva de la distribución de entrada.")
    with tabv:
        if total or fila is None:
            st.info("Elija una actividad en el selector de arriba para ver cómo la ventana de percentiles (centro más o menos el semiancho) "
                    "se traduce en un rango de horas dentro de la distribución de esa actividad.")
        else:
            # Se quitaron los dos gráficos (distribución acumulada y densidad); solo queda la tabla de ventanas.
            pres_v = [c_ for c_ in MWD_CLASES if (da['Calidad_Roca'] == c_).any()]
            Q = lambda p_: mwd_cuantil_actividad(fila['Tiempo'], fila['Distribucion'], float(np.clip(p_, 1e-6, 1 - 1e-6)))
            tabw = []
            for c in pres_v:
                rc = da[da['Calidad_Roca'] == c]
                if rc.empty:
                    continue
                cmin, cmax, cmed = float(rc['p_centro_MWD'].min()), float(rc['p_centro_MWD'].max()), float(rc['p_centro_MWD'].mean())
                plo, phi = max(cmin - sa, 1e-6), min(cmax + sa, 1 - 1e-6)
                xlo, xhi, xmid = Q(plo), Q(phi), Q(cmed)
                if None in (xlo, xhi, xmid):
                    continue
                dc_ = rc['Duration_Muestreada'].to_numpy(float)
                tabw.append({'Roca': c, f'Percentil central en el tiempo de {opcion} (mín. a máx.)': f"{cmin:.3f} a {cmax:.3f}",
                             f'Ventana de percentil en el tiempo de {opcion} (± {sa:.2f})': f"{plo:.3f} a {phi:.3f}",
                             f'Rango de duración de {opcion} en la ventana (h)': f"{xlo:.2f} a {xhi:.2f}",
                             f'Duración de {opcion} en el percentil central (h)': round(xmid, 2),
                             f'Duración de {opcion} obtenida en la simulación (h, mín. a máx.)': f"{dc_.min():.2f} a {dc_.max():.2f}"})
            if tabw:
                st.caption(f"Esta tabla es de la DURACIÓN de la actividad {opcion} (en horas), no del ROP. El ROP solo define la clase de roca; "
                           f"según la roca se elige una ventana de percentiles dentro de la distribución de tiempo de {opcion}, y de ahí sale su duración. "
                           "Los percentiles de la tabla son de esa distribución de tiempo (0 = la duración más corta posible, 1 = la más larga).")
                mostrar_tabla(pd.DataFrame(tabw), width='stretch', hide_index=True)
    with tab2:
        fig_b = go.Figure()
        for c in MWD_CLASES:
            sc = da[da['Calidad_Roca'] == c]
            if len(sc):
                txt = ("Sim " + sc['Simulation_ID'].astype(str) + " | " + sc['Resource_Destination_Tunnel'].astype(str) +
                       " | avance " + sc['Ciclo_Frente'].astype(str) + " | " + sc['Actividad'].astype(str))
                fig_b.add_trace(go.Box(y=sc[ycol], name=c, marker_color=MWD_COLORES[c], boxmean=True,
                                       boxpoints='all' if len(sc) <= 3000 else 'outliers', jitter=0.5, pointpos=-1.8,
                                       marker=dict(size=4, opacity=0.6), text=txt, hoverinfo='y+text'))
        ref = float(ref_fila.mean()) if not total and ref_fila.notna().any() else np.nan
        if total:
            fig_b.add_hline(y=0.0, line_dash="dash", line_color="#555", annotation_text="Mediana de entrada (0%)", annotation_position="top left")
        elif np.isfinite(ref):
            fig_b.add_hline(y=ref, line_dash="dash", line_color="#555",
                            annotation_text=f"Mediana de la distribución de entrada: {ref:.2f} h", annotation_position="top left")
        fig_b.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
                            xaxis_title="Clase de roca", yaxis_title=ylab)
        st.plotly_chart(fig_b, width='stretch', key=f"mwd_ver_box_{k}_{opcion}")
        st.caption("Cada caja es la duración obtenida en esa clase de roca (el rombo es la media). La línea punteada es la mediana de la "
                   "distribución de entrada: con regla inversa la roca mala queda por debajo y la roca buena por encima.")
        claves = ['Actividad', 'Calidad_Roca'] if total else ['Calidad_Roca']
        tab = (da.groupby(claves, observed=True)
               .agg(Ejecuciones=('Duration_Muestreada', 'count'), p_rop_min=('p_rop_eq', 'min'), p_rop_max=('p_rop_eq', 'max'),
                    p_tiempo_min=('p_tiempo_MWD', 'min'), p_tiempo_max=('p_tiempo_MWD', 'max'),
                    Duracion_min_h=('Duration_Muestreada', 'min'), Duracion_media_h=('Duration_Muestreada', 'mean'),
                    Duracion_max_h=('Duration_Muestreada', 'max'), Mediana_entrada_h=('Dur_Ref_Mediana_h', 'mean')).reset_index())
        if sim is None:
            gq = da.groupby(claves, observed=True)['Duration_Muestreada']
            qs = pd.DataFrame({'P10_h': gq.quantile(.1), 'P50_h': gq.quantile(.5), 'P90_h': gq.quantile(.9)}).reset_index()
            tab = tab.merge(qs, on=claves).rename(columns={'Duracion_media_h': 'Esperanza_h'})
            tab['Duracion_media_h'] = tab['Esperanza_h']
        tab['Variacion_vs_mediana_pct'] = np.where(tab['Mediana_entrada_h'] > 0,
                                                   (tab['Duracion_media_h'] / tab['Mediana_entrada_h'] - 1.0) * 100.0, np.nan)
        tab['Calidad_Roca'] = pd.Categorical(tab['Calidad_Roca'], categories=MWD_CLASES, ordered=True)
        mostrar_tabla(tab.drop(columns=['Duracion_media_h'] if sim is None else []).sort_values(claves).round(3), width='stretch', hide_index=True)
    if not hay_metro:
        return
    with tab3:
        sub = _mwd_subtramos(da, f"{k}_{opcion}_{len(da)}")
        if sub.empty:
            st.caption("No hay detalle por subtramo. Vuelva a simular para registrarlo.")
        else:
            _mwd_grafico_por_metro(sub, total, f"mwd_ver_pos_{k}_{opcion}", pct=sim is None)
            st.caption("Cada color de fondo es la roca de ese metro; cuando un ciclo cruza una interfaz (por ejemplo muy mala a muy buena) los "
                       "valores cambian en el metro exacto del cambio. Con regla inversa, al subir el p_rop baja el percentil de tiempo y el "
                       "la tasa de perforación es más alta (más m/h), y al revés. Con varias simulaciones se promedia por metro y se muestra la banda P10 a P90.")


def _mwd_box_rop(lg, k):
    """ROP medio de los frentes MWD sorteados, por clase (con los datos que salieron en la simulacion)."""
    if lg is None or len(lg) == 0:
        return
    st.markdown("#### ROP de los frentes MWD sorteados, por clase de roca",
                help="Cada punto es una referencia MWD sorteada en las simulaciones mostradas. La caja resume su ROP medio por clase.")
    pts = 'all' if len(lg) <= 3000 else 'outliers'
    st.plotly_chart(px.box(lg, x='Calidad_Roca', y='ROP_Medio', color='Calidad_Roca', color_discrete_map=MWD_COLORES,
                           category_orders={'Calidad_Roca': MWD_CLASES}, points=pts,
                           labels={'Calidad_Roca': 'Clase', 'ROP_Medio': 'ROP medio del frente sorteado'}),
                    width='stretch', key=f"mwd_res_boxrop_{k}")


def _mwd_kpis(lg):
    mostrar_tarjetas_metricas([
        ("Túneles con MWD", f"{lg['Tunel'].nunique()}", "", False),
        ("Tramos sorteados", f"{len(lg):,}", "", False, "referencias de roca sorteadas"),
        ("Frentes MWD distintos", f"{lg['Frente_MWD_Ref'].nunique():,}", "", False, "usados como referencia"),
        ("Simulaciones registradas", f"{lg['Simulacion'].nunique():,}", "", False)], ancho_max=250)


def mwd_mostrar_resultados(tab_geo, sim_global=None):
    """Pestaña de resultados. El alcance (todas las simulaciones o una específica) lo define el selector
    general de Resultados: con una simulación específica solo se muestra esa réplica."""
    with tab_geo:
        with seccion_menu('20b', 'Geología MWD',
                          ayuda="Qué roca se asignó, qué percentil de ROP produjo y cómo cambió la duración de las actividades con jumbo."):
            usada = st.session_state.get('mwd_config_usada')
            log = st.session_state.get('mwd_log_geologia')
            if not usada or log is None or len(log) == 0:
                st.info("Esta simulación no usó geología MWD. Configúrela en el Menú 10 (Restricciones Geológicas) y vuelva a simular.")
                return
            reglas = ", ".join(f"{k} ({v})" for k, v in usada['reglas'].items()) or "ninguna (solo etiqueta la roca)"
            st.caption(f"Semiancho {usada['semiancho']:.2f} | actividades afectadas: {reglas}. Las clases son relativas a la base "
                       "MWD cargada. Un ciclo que cruza dos rocas usa el promedio de sus duraciones ponderado por los metros de "
                       "cada roca (ver 'Mezcla_Roca').")
            run_id = st.session_state.get('run_id')
            df_tr = st.session_state.get('df_traza_eventos_cache')
            resultados = st.session_state.get('resultados') or {}

            if sim_global is not None:
                sim = int(sim_global)
                lg = log[log['Simulacion'] == sim]
                if len(lg) == 0:
                    st.info(f"La geología sorteada solo se registra para las primeras {MWD_MAX_SIMS_LOG} simulaciones. "
                            f"La simulación {sim} no tiene registro; elija una de las primeras {MWD_MAX_SIMS_LOG}.")
                    return
                st.markdown(f"### Simulación {sim}")
                _mwd_kpis(lg)
                _mwd_ui_perfil_resultados(log, sim)
                if resultados:
                    av = [{'Túnel': t, 'Avance final (m)': round(float(s[min(sim - 1, len(s) - 1)]), 2)}
                          for t, s in resultados.items() if len(s) and t in set(lg['Tunel'].astype(str))]
                    if av:
                        st.markdown("#### Avance final de esta simulación")
                        mostrar_tabla(pd.DataFrame(av), width='stretch', hide_index=True)
                _mwd_box_rop(lg, f"s{sim}")
                _mwd_efecto_actividades(df_tr, usada, f"s_{sim}", sim=sim)
                st.markdown("---")
                st.markdown("### Resultados por avance")
                tsel = st.selectbox("Túnel", sorted(lg['Tunel'].astype(str).unique()), key="mwd_res_sim_tunel")
                _mwd_seccion_ciclo(df_tr, log, usada, tsel, sim, f"s{sim}_{tsel}")
                return

            # ---- Todas las simulaciones: resumen global y, abajo, resultados por tunel ----
            tuneles = sorted(log['Tunel'].astype(str).unique())
            st.markdown("### Resumen global (todas las simulaciones)")
            _mwd_kpis(log)
            st.markdown("#### Clase de roca por túnel (% de los metros)")
            pct = _mwd_pct_clases(log)
            _w = log.groupby('Calidad_Roca')['Largo_m'].sum().reindex(MWD_CLASES).fillna(0)
            _fila = (_w / _w.sum() * 100.0).to_frame().T
            _fila.index = [f"TODOS ({len(tuneles)})"]
            _fila.index.name = 'Tunel'
            _pp = pd.concat([_fila, pct.reindex(tuneles)])
            _long = _pp.reset_index().melt(id_vars='Tunel', var_name='Calidad_Roca', value_name='Porcentaje')
            _fg = px.bar(_long, y='Tunel', x='Porcentaje', color='Calidad_Roca', orientation='h',
                         color_discrete_map=MWD_COLORES, category_orders={'Calidad_Roca': MWD_CLASES, 'Tunel': list(_pp.index)},
                         labels={'Porcentaje': '% de los metros', 'Tunel': '', 'Calidad_Roca': 'Clase'})
            _fg.update_layout(height=22 * len(_pp) + 90, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(range=[0, 100]),
                              yaxis=dict(autorange='reversed'), bargap=0.25)
            if len(_pp) > 20:
                with st.container(height=460):
                    st.plotly_chart(_fg, width='stretch', key="mwd_res_clases_g")
            else:
                st.plotly_chart(_fg, width='stretch', key="mwd_res_clases_g")
            # ---- Geología sorteada en 3D: todos los túneles, cada tramo con su roca (no la proporción) ----
            st.markdown("#### Geología sorteada en 3D (todos los túneles)")
            # La geología es la misma en todas las simulaciones: se dibuja una sola vez (sin selector de simulación).
            _s3d = int(log['Simulacion'].min())
            _f3d = _mwd_fig_3d(mwd_perfiles_desde_log(log, _s3d), "Geología sorteada (igual en todas las simulaciones)", altura=640)
            if _f3d is not None:
                st.plotly_chart(_f3d, width='stretch', key="mwd_g3d_fig")
                st.caption("Rojo = muy mala, naranja = mala, amarillo = regular, verde claro = buena, verde = muy buena. "
                           "Cada tramo del túnel tiene la roca asignada; es la misma en todas las simulaciones.")
            tab_pct = pct.round(1).reset_index()
            tab_pct['p_rop medio'] = log.groupby('Tunel')['p_rop'].mean().reindex(tab_pct['Tunel']).round(3).to_numpy()
            with st.expander(f"Ver la tabla de cada túnel ({len(tuneles)})"):
                mostrar_tabla(tab_pct, width='stretch', hide_index=True)
            _mwd_box_rop(log, "g")
            _mwd_efecto_actividades(df_tr, usada, "g")
            with st.expander("Detalle por tramo (túnel, simulación, referencia MWD usada)"):
                mostrar_tabla(log.head(5000), width='stretch', hide_index=True)
                st.download_button("Descargar detalle (CSV)", log.to_csv(index=False).encode('utf-8-sig'),
                                   file_name="geologia_mwd_por_tramo.csv", mime="text/csv", key="mwd_dl_log")

            st.markdown("---")
            st.markdown("### Resultados por túnel")
            t = st.selectbox("Túnel", tuneles, key="mwd_res_tunel_sel")
            lt = log[log['Tunel'].astype(str) == t]
            serie = resultados.get(t)
            tarj = [("Modo", str(lt['Modo'].iloc[0]), "", False), ("Tramos sorteados", f"{len(lt):,}", "", False),
                    ("p_rop medio", f"{lt['p_rop'].mean():.3f}", "", False)]
            if serie is not None and len(serie):
                tarj.append(("Avance final medio", f"{float(np.mean(serie)):.1f}", "m", True))
            mostrar_tarjetas_metricas(tarj, ancho_max=250)
            st.markdown(f"#### Clase de roca en {t} (% de los metros)")
            pt = _mwd_pct_clases(lt).iloc[0].reset_index()
            pt.columns = ['Calidad_Roca', 'Porcentaje']
            st.plotly_chart(px.bar(pt, x='Calidad_Roca', y='Porcentaje', color='Calidad_Roca', color_discrete_map=MWD_COLORES,
                                   category_orders={'Calidad_Roca': MWD_CLASES},
                                   labels={'Porcentaje': '% de los metros', 'Calidad_Roca': 'Clase'}),
                            width='stretch', key="mwd_res_clases_t")
            _mwd_box_rop(lt, f"t_{t}")
            _mwd_efecto_actividades(df_tr, usada, f"t_{t}", tunel=t)
            with st.expander("Detalle de tramos de este túnel"):
                mostrar_tabla(lt.head(5000), width='stretch', hide_index=True)

            st.markdown("---")
            st.markdown(f"### Resultados por avance de {t}")
            _mwd_seccion_ciclo(df_tr, log, usada, t, None, f"t_{t}")


@st.cache_data(show_spinner=False)
def _mwd_subtramos_cache(_da, clave):
    """Subtramos de roca de cada ciclo (uno por roca cruzada), con p_rop, p de tiempo, horas y tasa de perforación (m/h)."""
    import json as _json
    filas = []
    if 'Tramos_MWD' not in _da.columns:
        return pd.DataFrame()
    for r in _da[['Simulation_ID', 'Resource_Destination_Tunnel', 'Ciclo_Frente', 'Actividad', 'Tramos_MWD',
                  'Dur_Ref_Mediana_h']].itertuples(index=False):
        if not isinstance(r.Tramos_MWD, str):
            continue
        trs = _json.loads(r.Tramos_MWD)
        cub = sum(t_[1] - t_[0] for t_ in trs)
        for t_ in trs:
            d0, d1, c, pr, pt, v, h = t_[:7]
            L = t_[7] if len(t_) > 7 and t_[7] else cub
            filas.append((r.Simulation_ID, str(r.Resource_Destination_Tunnel), r.Ciclo_Frente, r.Actividad, d0, d1, c, pr, pt, v, h,
                          r.Dur_Ref_Mediana_h, L))
    df = pd.DataFrame(filas, columns=['sim', 'tunel', 'ciclo', 'actividad', 'd0', 'd1', 'clase', 'p_rop', 'p_t', 'v', 'horas', 'ref', 'L'])
    if df.empty:
        return df
    df['largo'] = df['d1'] - df['d0']
    df['ritmo'] = np.where((df['L'] > 0) & (df['v'] > 0), df['L'] / df['v'], np.nan)  # tasa de perforación (ROP) en m/h: metros NOMINALES del ciclo / duración
    df['rel'] = np.where(df['ref'] > 0, (df['v'] / df['ref'] - 1.0) * 100.0, np.nan)
    return df


def _mwd_subtramos(da, clave):
    return _mwd_subtramos_cache(da, (clave, st.session_state.get('run_id'), len(da)))


def _mwd_grafico_por_metro(sub, total, key, pct=True):
    """Grafico por metro: p_rop, p de tiempo y ritmo de perforacion por subtramo de roca (escalones en cada cambio)."""
    from plotly.subplots import make_subplots
    ycol, ylab = ('rel', 'Variación de la duración respecto a la mediana (%)') if total else ('ritmo', 'Tasa de perforación (m/h)')
    ELEM_M = ["Fondo de roca", "p_rop de la roca", "Percentil de tiempo sorteado", "Ritmo de perforación"] + (["Banda P10 a P90 del ritmo", "Banda P10 a P90 de los percentiles"] if pct else [])
    elems = st.multiselect("Elementos a mostrar", ELEM_M, default=ELEM_M, key=f"{key}_el2")
    lo_m, hi_m = float(sub['d0'].min()), float(sub['d1'].max())
    if hi_m - lo_m <= 0:
        st.caption("Sin rango de metros para graficar.")
        return
    n_cel = int(min(300, max(8, np.ceil((hi_m - lo_m) / 0.25))))
    edges = np.linspace(lo_m, hi_m, n_cel + 1)
    cent = (edges[:-1] + edges[1:]) / 2.0
    d0, d1 = sub['d0'].to_numpy(float), sub['d1'].to_numpy(float)
    prop, pt, yv = sub['p_rop'].to_numpy(float), sub['p_t'].to_numpy(float), sub[ycol].to_numpy(float)
    cls = pd.Categorical(sub['clase'], categories=MWD_CLASES).codes
    out = {k_: np.full(n_cel, np.nan) for k_ in ('p_rop', 'p_t', 'y', 'lo', 'hi', 'n', 'pr_lo', 'pr_hi', 'pt_lo', 'pt_hi')}
    clase = [None] * n_cel
    for j, c in enumerate(cent):
        m = (d0 <= c) & (d1 > c)
        if not m.any():
            continue
        out['p_rop'][j], out['p_t'][j], out['y'][j] = prop[m].mean(), pt[m].mean(), np.nanmean(yv[m])
        out['lo'][j], out['hi'][j] = np.nanpercentile(yv[m], 10), np.nanpercentile(yv[m], 90)
        out['pr_lo'][j], out['pr_hi'][j] = np.nanpercentile(prop[m], 10), np.nanpercentile(prop[m], 90)
        out['pt_lo'][j], out['pt_hi'][j] = np.nanpercentile(pt[m], 10), np.nanpercentile(pt[m], 90)
        out['n'][j] = m.sum()
        cc = np.bincount(cls[m][cls[m] >= 0], minlength=len(MWD_CLASES))
        clase[j] = MWD_CLASES[int(cc.argmax())] if cc.sum() else None

    def _rgba(hexc, a):
        return _mwd_rgba(hexc, a)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    j = 0
    while j < n_cel:
        if clase[j] is None:
            j += 1
            continue
        k_ = j
        while k_ + 1 < n_cel and clase[k_ + 1] == clase[j]:
            k_ += 1
        fig.add_shape(visible=('Fondo de roca' in elems), type='rect', xref='x', yref='paper', x0=edges[j], x1=edges[k_ + 1], y0=0, y1=1, layer='below', line_width=0,
                      fillcolor=_rgba(MWD_COLORES[clase[j]], 0.30))
        j = k_ + 1
    for c in MWD_CLASES:
        if c in clase:
            fig.add_trace(go.Scatter(visible=('Fondo de roca' in elems), x=[None], y=[None], mode='markers', name=f"Roca {c}",
                                     marker=dict(symbol='square', size=12, color=_rgba(MWD_COLORES[c], 0.55))), secondary_y=False)
    xs = np.repeat(edges, 2)[1:-1]
    def _st(v):
        return np.repeat(v, 2)
    if pct and 'Banda P10 a P90 del ritmo' in elems:
        fig.add_trace(go.Scatter(x=xs, y=_st(out['hi']), mode='lines', line=dict(width=0), hoverinfo='skip', showlegend=False), secondary_y=True)
        fig.add_trace(go.Scatter(x=xs, y=_st(out['lo']), mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(60,60,60,0.12)',
                                 name='Ritmo P10 a P90' + (' (% vs mediana)' if total else ' (m/h)'), hoverinfo='skip'), secondary_y=True)
    if pct and 'Banda P10 a P90 de los percentiles' in elems:
        for nom_, lo_, hi_, col_, rgb_ in (("p_rop de la roca", 'pr_lo', 'pr_hi', '#1565c0', 'rgba(21,101,192,0.15)'),
                                           ("Percentil de tiempo sorteado", 'pt_lo', 'pt_hi', '#ef6c00', 'rgba(239,108,0,0.15)')):
            if nom_ not in elems:
                continue
            fig.add_trace(go.Scatter(x=xs, y=_st(out[hi_]), mode='lines', line=dict(width=0), hoverinfo='skip', showlegend=False), secondary_y=False)
            fig.add_trace(go.Scatter(x=xs, y=_st(out[lo_]), mode='lines', line=dict(width=0), fill='tonexty', fillcolor=rgb_,
                                     name=nom_ + ': P10 a P90 entre simulaciones', hoverinfo='skip'), secondary_y=False)
    tx = [f"Metro {c_:.2f}<br>Roca: {k2}<br>Subtramos: {int(n) if np.isfinite(n) else 0}" for c_, k2, n in zip(cent, clase, out['n'])]
    for nombre, v_, col, sec in (("p_rop de la roca", out['p_rop'], '#1565c0', False), ("Percentil de tiempo sorteado", out['p_t'], '#ef6c00', False),
                                 (("Ritmo esperanza" if pct else "Ritmo") + (" (m/h)" if not total else " (% vs mediana)"), out['y'], '#212121', True)):
        clave_el = 'Ritmo de perforación' if sec else ('p_rop de la roca' if nombre.startswith('p_rop') else 'Percentil de tiempo sorteado')
        if clave_el not in elems:
            continue
        fig.add_trace(go.Scatter(x=xs, y=_st(v_), mode='lines', name=nombre, line=dict(color=col, width=2.5 if sec else 2),
                                 line_shape='linear', hoverinfo='skip'), secondary_y=sec)
        fig.add_trace(go.Scatter(x=cent, y=v_, mode='markers', name=nombre, showlegend=False, marker=dict(color=col, size=4),
                                 text=tx, hovertemplate="%{text}<br>" + nombre + ": %{y:.3f}<extra></extra>"), secondary_y=sec)
    fig.update_xaxes(title_text="Metro de avance del túnel")
    fig.update_yaxes(title_text="Percentil (0 a 1)", range=[0, 1], secondary_y=False)
    fig.update_yaxes(title_text=ylab, secondary_y=True)
    fig.update_layout(height=470, margin=dict(l=10, r=10, t=30, b=10), hovermode='closest', plot_bgcolor='rgba(0,0,0,0)',
                      legend=dict(orientation='h', y=-0.2, title_text=''))
    _mwd_leyenda_abajo(fig)
    st.plotly_chart(fig, width='stretch', key=key)


def _mwd_resumen_avances(d, sub, k, sim=None):
    """Todos los avances de un tunel, de a una metrica por vez. Todas las simulaciones: esperanza y percentiles entre
    simulaciones. Una simulacion: valor directo de esa replica."""
    from plotly.subplots import make_subplots
    todas = sim is None
    ex = (sub.assign(w_prop=sub['p_rop'] * sub['largo'], w_pt=sub['p_t'] * sub['largo'])
          .groupby(['sim', 'ciclo', 'actividad'])
          .agg(largo=('largo', 'sum'), horas=('horas', 'sum'), w_prop=('w_prop', 'sum'), w_pt=('w_pt', 'sum')).reset_index())
    ex['dur'] = ex['horas']
    ex['ritmo'] = ex['largo'] / sub.assign(w_h=sub['largo'] / sub['ritmo']).groupby(['sim', 'ciclo', 'actividad'])['w_h'].sum().to_numpy()  # m/h = metros / horas
    ex['p_rop'] = ex['w_prop'] / ex['largo']
    ex['p_t'] = ex['w_pt'] / ex['largo']
    mez, dom, rng_m = {}, {}, {}
    for c_, gg in sub.groupby('ciclo'):
        m_ = gg.groupby('clase')['largo'].sum().reindex(MWD_CLASES).dropna()
        mez[c_] = " / ".join(f"{cl} {v / m_.sum() * 100:.0f}%" for cl, v in m_.items())
        dom[c_] = m_.idxmax()
        rng_m[c_] = (gg['d0'].min(), gg['d1'].max())
    met = st.radio("Qué ver en cada avance", ["Duración de la actividad (h)", "Tasa de perforación (m/h)", "Percentiles (p_rop y p de tiempo)"],
                   horizontal=True, key=f"mwd_av_met_{k}",
                   help="Se muestra una métrica a la vez. " + ("Con todas las simulaciones se grafican la esperanza y la banda P10 a P90 entre simulaciones."
                                                               if todas else "Son los valores directos de esta simulación."))
    acts_all = sorted(ex['actividad'].unique())
    f1, f2 = st.columns([2, 3])
    act_sel = f1.multiselect("Actividades a mostrar", acts_all, default=acts_all, key=f"mwd_av_act_{k}",
                             help="Solo aplica a duración y ritmo; los percentiles son de la roca.")
    ELEM_A = ["Fondo de roca", "Esperanza", "P50 (mediana)", "Banda P10 a P90"] if todas else ["Fondo de roca", "Valores"]
    elem = f2.multiselect("Elementos a mostrar", ELEM_A, default=ELEM_A, key=f"mwd_av_el_{k}_{met}")
    paleta = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#8c564b', '#17becf', '#e377c2', '#7f7f7f']

    def _rgba(hexc, a_):
        return _mwd_rgba(hexc, a_)

    def _est(g, col):
        q = g[col]
        return pd.DataFrame({'Esperanza': q.mean(), 'P10': q.quantile(.1), 'P50': q.quantile(.5), 'P90': q.quantile(.9),
                             'Mín': q.min(), 'Máx': q.max()}) if todas else pd.DataFrame({'Valor': q.mean()})
    fig = go.Figure()
    for c_ in sorted(dom):
        fig.add_shape(visible=('Fondo de roca' in elem), type='rect', xref='x', yref='paper', x0=c_ - 0.5, x1=c_ + 0.5, y0=0, y1=1, layer='below', line_width=0,
                      fillcolor=_rgba(MWD_COLORES[dom[c_]], 0.28))
    for c in MWD_CLASES:
        if c in dom.values():
            fig.add_trace(go.Scatter(visible=('Fondo de roca' in elem), x=[None], y=[None], mode='markers', name=f"Roca {c}",
                                     marker=dict(symbol='square', size=12, color=_rgba(MWD_COLORES[c], 0.55))))
    if met.startswith("Percentiles"):
        series = [("p_rop de la roca", 'p_rop', '#1565c0', ex), ("Percentil de tiempo sorteado", 'p_t', '#ef6c00', ex)]
        unidad, ytitle, rango = "", "Percentil (0 a 1)", [0, 1]
    else:
        col_m, unidad = ('dur', ' h') if met.startswith("Duración") else ('ritmo', ' m/h')
        ytitle, rango = met, None
        series = [(a_, col_m, paleta[i % len(paleta)], ex[ex['actividad'] == a_]) for i, a_ in enumerate(sorted(ex['actividad'].unique())) if a_ in act_sel]
    tablas = []
    for nombre, col, color, dat in series:
        g = dat.groupby('ciclo')
        es = _est(g, col)
        xv = es.index.to_numpy()
        if todas:
            fig.add_trace(go.Scatter(visible=('Banda P10 a P90' in elem), x=xv, y=es['P90'], mode='lines', line=dict(width=0), hoverinfo='skip', showlegend=False))
            fig.add_trace(go.Scatter(visible=('Banda P10 a P90' in elem), x=xv, y=es['P10'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor=_rgba(color, 0.18),
                                     name=f"{nombre}: P10 a P90", hoverinfo='skip'))
            fig.add_trace(go.Scatter(visible=('P50 (mediana)' in elem), x=xv, y=es['P50'], mode='lines', line=dict(color=color, width=1.5, dash='dot'), name=f"{nombre}: P50"))
            fig.add_trace(go.Scatter(visible=('Esperanza' in elem), x=xv, y=es['Esperanza'], mode='lines+markers', line=dict(color=color, width=3), marker=dict(size=8),
                                     name=f"{nombre}: esperanza",
                                     customdata=np.stack([es['P10'], es['P50'], es['P90'], es['Mín'], es['Máx']], axis=1),
                                     hovertemplate=f"{nombre}<br>Avance %{{x}}<br>Esperanza %{{y:.3f}}{unidad}<br>P10 %{{customdata[0]:.3f}} | "
                                                   "P50 %{customdata[1]:.3f} | P90 %{customdata[2]:.3f}<br>Mín %{customdata[3]:.3f} | "
                                                   "Máx %{customdata[4]:.3f}<extra></extra>"))
        else:
            fig.add_trace(go.Scatter(visible=('Valores' in elem), x=xv, y=es['Valor'], mode='lines+markers', line=dict(color=color, width=3), marker=dict(size=9),
                                     name=nombre, hovertemplate=f"{nombre}<br>Avance %{{x}}<br>Valor %{{y:.3f}}{unidad}<extra></extra>"))
        t_ = es.reset_index().rename(columns={'ciclo': 'Avance'})
        t_.insert(1, 'Serie', nombre)
        tablas.append(t_)
    fig.update_xaxes(title_text="Avance del túnel (N°)", dtick=1 if len(dom) <= 40 else None)
    fig.update_yaxes(title_text=ytitle, range=rango)
    fig.update_layout(height=430, hovermode='x unified', margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor='rgba(0,0,0,0)',
                      legend=dict(orientation='h', y=-0.25, title_text=''))
    _mwd_leyenda_abajo(fig)
    st.plotly_chart(fig, width='stretch', key=f"mwd_av_res_{k}_{met}")
    st.caption("El fondo es la roca predominante de cada avance (cuando un avance cruza una interfaz, la tabla muestra el % de cada roca). "
               + ("Línea gruesa: esperanza entre simulaciones; punteada: mediana (P50); banda: P10 a P90." if todas else
                  "Valores de esta simulación, sin percentiles."))
    tab = pd.concat(tablas, ignore_index=True)
    tab.insert(1, 'Desde_m', tab['Avance'].map(lambda c_: rng_m[c_][0]))
    tab.insert(2, 'Hasta_m', tab['Avance'].map(lambda c_: rng_m[c_][1]))
    tab.insert(3, 'Roca (% de los metros)', tab['Avance'].map(mez))
    mostrar_tabla(tab.round(3), width='stretch', hide_index=True)


def _mwd_seccion_ciclo(df_tr, log, usada, tunel, sim, k):
    """Resultados por avance de un tunel (de una simulacion o de todas): resumen de todos los avances y detalle de uno."""
    if df_tr is None or 'Regla_MWD' not in df_tr.columns:
        return
    d = df_tr[(df_tr['type'] == 'activity') & df_tr['Regla_MWD'].notna() &
              (df_tr['Resource_Destination_Tunnel'].astype(str) == str(tunel))]
    if sim is not None:
        d = d[d['Simulation_ID'] == sim]
    if d.empty:
        st.caption("Este túnel no tiene avances con actividades afectadas por MWD en este alcance.")
        return
    if 'Tramos_MWD' not in d.columns or not d['Tramos_MWD'].notna().any():
        st.warning("Esta corrida no registró el detalle por avance. Vuelva a simular con esta versión del programa.")
        return
    sub_all = _mwd_subtramos(d, f"{k}_all")
    if sub_all.empty:
        st.warning("No se pudo leer el detalle por avance de esta corrida. Vuelva a simular.")
        return
    st.markdown("#### Todos los avances del túnel")
    _mwd_resumen_avances(d, sub_all, k, sim)
    ciclos = sorted(int(c) for c in d['Ciclo_Frente'].dropna().unique())
    c = st.selectbox("Avance a detallar", ciclos, key=f"mwd_ciclo_sel_{k}",
                     help="N° de avance del propio túnel (avance 1, 2, 3...). Con todas las simulaciones se resumen las ejecuciones de ese avance en todas ellas.")
    dc = d[d['Ciclo_Frente'] == c]
    sub = sub_all[sub_all['ciclo'] == c]
    if sub.empty:
        st.caption("Este avance no tiene detalle de subtramos.")
        return
    a, b = float(sub['d0'].min()), float(sub['d1'].max())
    st.markdown(f"### Detalle del avance {c}")
    mostrar_tarjetas_metricas([("Avance", f"{c}", "", True), ("Metros del avance", f"{a:.1f} a {b:.1f}", "m", False),
                               ("Rocas cruzadas", f"{sub['clase'].nunique()}", "", False)], ancho_max=250)
    pc = sub.groupby('clase')['largo'].sum().reindex(MWD_CLASES).dropna()
    st.markdown(f"#### Roca dentro del avance {c} (% de los metros)")
    figc = px.bar(pc.div(pc.sum()).mul(100).reset_index().rename(columns={'clase': 'Clase', 'largo': 'Porcentaje'}),
                  x='Clase', y='Porcentaje', color='Clase', color_discrete_map=MWD_COLORES, category_orders={'Clase': MWD_CLASES})
    figc.update_layout(height=300, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(figc, width='stretch', key=f"mwd_ciclo_pct_{k}_{c}")
    if log is not None and len(log):
        lg = log[(log['Tunel'].astype(str) == str(tunel)) & (log['Tramo_Hasta_m'] > a) & (log['Tramo_Desde_m'] < b)]
        if sim is not None:
            lg = lg[lg['Simulacion'] == sim]
        _mwd_box_rop(lg, f"c_{k}_{c}")
    _mwd_efecto_actividades(df_tr, usada, f"c_{k}_{c}", tunel=tunel, sim=sim, ciclo=c)
    with st.expander("Detalle de subtramos de este avance"):
        mostrar_tabla(sub.drop(columns=['ref', 'L']).sort_values(['sim', 'actividad', 'd0']).round(3).head(5000), width='stretch', hide_index=True)
def simular_avance_con_transporte(actividades_nivel, frentes_info, recursos_config, 
                                   tiempo_limite, metros_avance, n_simulaciones, 
                                   velocidades_config, demoras_constantes, demoras_por_turno, 
                                   sistema_turnos, restricciones_geologicas, fallas_equipos,
                                   plan_recursos_turno=None, progress_callback=None,
                                   bloqueos_tuneles=None, mantenciones_programadas=None,
                                   demoras_cambio_turno=None, config_mwd=None):
    """
    Simulación con gestión de eventos discretos: cada actividad libera su recurso inmediatamente.

    bloqueos_tuneles: dict opcional {frente: [(inicio_h, fin_h), ...]} (ver
    calcular_bloqueos_tuneles). Mientras el reloj de la simulación caiga dentro de una
    de estas ventanas para un frente, ese frente queda con avance 0 y no se le asigna
    ningún recurso — bloqueo TOTAL, distinto de 'demoras_constantes'/'demoras_por_turno'
    (que solo retrasan el inicio del ciclo unas horas pero el frente sigue avanzando).
    """
    if plan_recursos_turno is None:
        plan_recursos_turno = []
    if bloqueos_tuneles is None:
        bloqueos_tuneles = {}
    if mantenciones_programadas is None:
        mantenciones_programadas = []
    
    # Determinar duración del turno
    if '12x2' in sistema_turnos:
        duracion_turno = 12.0
    elif '8x3' in sistema_turnos:
        duracion_turno = 8.0
    else:
        duracion_turno = 24.0

    # Los intervalos de bloqueo total ya vienen en horas absolutas de simulación (ver
    # calcular_bloqueos_tuneles), asi que solo se recortan al horizonte real que se esta
    # simulando ahora (por si tiempo_limite cambio despues de haberlos configurado).
    bloqueos_totales_frente = {
        frente: [(ini, min(fin, tiempo_limite)) for ini, fin in intervalos if min(fin, tiempo_limite) > ini]
        for frente, intervalos in bloqueos_tuneles.items()
    }
    bloqueos_totales_frente = {f: iv for f, iv in bloqueos_totales_frente.items() if iv}

    n_frentes = len(frentes_info)
    frentes_nombres = list(frentes_info.keys())
    
    # Pre-procesar restricciones geológicas para búsqueda rápida
    mapa_restriccion_geologica = {}
    for frente in frentes_nombres:
        for res_geo in restricciones_geologicas:
            if frente in res_geo.get('frentes_aplicables', []):
                mapa_restriccion_geologica[frente] = res_geo
                break  # Aplicar solo la primera restricción encontrada

    # Pre-procesar fallas de equipos para búsqueda rápida
    mapa_fallas_equipos = {}
    for falla_config in fallas_equipos:
        for frente in falla_config.get('frentes_aplicables', []):
            mapa_fallas_equipos[frente] = falla_config

    def metros_ciclo(frente_nombre):
        return avance_efectivo_frente(frente_nombre, metros_avance, frentes_info, mapa_restriccion_geologica)

    resultados = {frente: [] for frente in frentes_nombres}
    traza_eventos = []
    # Avance REAL acumulado por turno, capturado en vivo desde avances_frentes durante la
    # simulacion (misma fuente que "resultados"). Reemplaza el calculo posterior por conteo
    # de ciclos en la traza de eventos, que podia mostrar avance donde el Avance Final real
    # quedo en 0 (ciclos truncados por tiempo_limite o por bloqueo de recurso).
    n_turnos_total = math.ceil(tiempo_limite / duracion_turno)
    avance_por_turno_resultado = {
        frente: {turno: [] for turno in range(1, n_turnos_total + 1)} for frente in frentes_nombres
    }
    
    # Pre-calcular velocidades por recurso. Cada recurso es independiente de los demás
    # (misma logica, distintos parametros), asi que el muestreo se puede paralelizar con
    # hilos: scipy/numpy liberan el GIL durante buena parte del computo vectorizado, asi
    # que hilos si ayudan aca (a diferencia de codigo Python puro CPU-bound). Para que sea
    # SEGURO, cada hilo recibe su propio numpy.random.Generator independiente (via
    # SeedSequence.spawn) en vez de compartir el generador global de NumPy: usar el
    # generador global desde varios hilos a la vez no es thread-safe y podria producir
    # muestras correlacionadas o no reproducibles entre corridas. Antes de intentarlo se
    # valida que la version de scipy instalada acepte un Generator en random_state (todas
    # las de los ultimos años lo aceptan); si no, o si cualquier cosa con los hilos falla,
    # se cae directo al mismo loop secuencial de siempre (generador global, sin cambios de
    # comportamiento).
    num_precalc = max(n_simulaciones * n_frentes * 100, 50000)
    velocidades_precalculadas = {}
    # Aplana las velocidades: clave = recurso (velocidad comun) o (recurso, id_unidad) cuando el
    # recurso tiene una distribucion propia por unidad (Menu 6, 'Definir velocidad por unidad').
    # Las unidades sin entrada propia (ej. incorporadas despues por el plan de flota) usan la del recurso.
    velocidades_flat = {}
    for _rec, _cfg in velocidades_config.items():
        velocidades_flat[_rec] = _cfg
        for _uid, _cfg_u in (_cfg.get('por_unidad') or {}).items():
            velocidades_flat[(_rec, int(_uid))] = _cfg_u
    recursos_a_precalcular = list(velocidades_flat.items())

    _random_state_soportado = True
    try:
        stats.norm.rvs(loc=0, scale=1, size=1, random_state=np.random.default_rng(0))
    except Exception:
        _random_state_soportado = False

    if _random_state_soportado and len(recursos_a_precalcular) > 1:
        try:
            semillas = np.random.SeedSequence().spawn(len(recursos_a_precalcular))
            with ThreadPoolExecutor(max_workers=min(8, len(recursos_a_precalcular))) as executor:
                futuros = {
                    executor.submit(
                        generar_velocidad_batch, vel_config['params'], vel_config['dist'],
                        num_precalc, np.random.default_rng(semilla)
                    ): recurso
                    for (recurso, vel_config), semilla in zip(recursos_a_precalcular, semillas)
                }
                for futuro, recurso in futuros.items():
                    velocidades_precalculadas[recurso] = futuro.result()
        except Exception:
            velocidades_precalculadas = {}  # descarta cualquier resultado parcial y cae a secuencial

    if not velocidades_precalculadas and recursos_a_precalcular:
        # Fallback secuencial: comportamiento identico al original (generador global).
        for recurso, vel_config in velocidades_flat.items():
            velocidades_precalculadas[recurso] = generar_velocidad_batch(vel_config['params'], vel_config['dist'], num_precalc)

    actividades_filtradas = actividades_nivel[actividades_nivel['Actividad'].notna()].copy()
    recursos_necesarios = list(actividades_filtradas['Recurso'].dropna().unique())
    
    # Obtener secuencias ordenadas
    secuencias_ordenadas = sorted(actividades_filtradas['Secuencia'].unique())

    # Pre-calcular, para cada combinacion (secuencia, frente), la lista de actividades
    # aplicables. ANTES esto se recalculaba con un DataFrame.apply(axis=1) dentro del
    # handler del evento 'start_sequence' (ver mas abajo, donde se asigna 'acts_secuencia'):
    # se ejecutaba una vez por cada secuencia, de cada ciclo, de cada frente, de CADA UNA
    # de las n_simulaciones del Monte Carlo -- miles de recalculos identicos por corrida --
    # para un resultado que es 100% deterministico (is_applicable() solo depende de
    # 'secuencia_num' y 'frente', que no cambian entre simulaciones; 'actividades_filtradas'
    # ya esta fijo antes de este punto). Se precalcula UNA sola vez aqui, con la MISMA
    # funcion de aplicabilidad exacta de antes (ningun cambio de criterio), y se guarda
    # como lista de dicts (to_dict('records')) en vez de DataFrame para tambien evitar el
    # costo de iterrows() en el loop de eventos.
    def _es_actividad_aplicable(row, frente_actual):
        if row.get('Es_Personalizada', False):
            # Solo aplicable si el frente está en la lista de frentes aplicables
            return frente_actual in row.get('Frentes_Aplicables', [])
        else:
            # Las actividades no personalizadas se aplican a todos los frentes por defecto
            return True

    acts_por_secuencia_frente = {}
    for secuencia_num in secuencias_ordenadas:
        acts_secuencia_base_precalc = actividades_filtradas[actividades_filtradas['Secuencia'] == secuencia_num]
        for frente in frentes_nombres:
            acts_filtradas_precalc = acts_secuencia_base_precalc[
                acts_secuencia_base_precalc.apply(lambda row: _es_actividad_aplicable(row, frente), axis=1)
            ]
            acts_por_secuencia_frente[(secuencia_num, frente)] = acts_filtradas_precalc.to_dict('records')

    # Pre-calcular ventanas de actividad (activa_desde/activa_hasta) por recurso segun
    # el plan de orquestacion de flota por turno. Es igual para todas las simulaciones
    # (determinístico), asi que se calcula una sola vez fuera del loop de Monte Carlo.
    ventanas_por_recurso = {}
    registro_cambios_flota = []
    for recurso in recursos_necesarios:
        if recurso in recursos_config:
            # math.ceil() (no int()): ver nota en calcular_ventanas_flota sobre
            # cantidades fraccionarias (ej. Roboshot = 0.5) truncadas a 0.
            cantidad_base = math.ceil(float(recursos_config[recurso]['cantidad']))
            ventanas, cambios_recurso = calcular_ventanas_flota(
                recurso, cantidad_base, plan_recursos_turno, duracion_turno
            )
            ventanas_por_recurso[recurso] = ventanas
            registro_cambios_flota.extend(cambios_recurso)
    
    # Cada cuantas simulaciones se notifica el progreso al callback. Se evita notificar en
    # cada iteracion cuando n_simulaciones es grande (ej. 1000+) para no saturar la UI con
    # actualizaciones innecesarias; con n_simulaciones chico (ej. 10) notifica cada una.
    intervalo_notificacion_progreso = max(1, n_simulaciones // 100)

    # Geologia desde MWD (opcional): preparacion unica y estado aleatorio independiente por replica.
    _mwd_prep = mwd_preparar(config_mwd, frentes_nombres) if (config_mwd and config_mwd.get('activo')) else None
    _mwd_ss = np.random.SeedSequence(config_mwd.get('semilla')) if _mwd_prep else None
    _mwd_rt = None
    for sim in range(n_simulaciones):
        _mwd_rt = mwd_runtime_sim(_mwd_prep, _mwd_ss, frentes_info, sim) if _mwd_prep else None
        # Inicializar recursos segun sus ventanas de actividad (cantidad base +/- cambios de plan)
        instancias_recursos = {}
        for recurso in recursos_necesarios:
            if recurso in ventanas_por_recurso and ventanas_por_recurso[recurso]:
                instancias_recursos[recurso] = []
                for ventana in ventanas_por_recurso[recurso]:
                    instancias_recursos[recurso].append({
                        'id': ventana['id'], 
                        'x': frentes_info[frentes_nombres[0]]['xi'],
                        'y': frentes_info[frentes_nombres[0]]['yi'],
                        'frente_actual': frentes_nombres[0], 
                        'disponible_en': max(0.0, ventana['activa_desde']),
                        'activa_desde': ventana['activa_desde'],
                        'activa_hasta': ventana['activa_hasta'],
                        'ventanas_mantenimiento': [],  # se completa mas abajo (Menu 12)
                        'mantenciones_info': []        # (ini, fin, nombre) de cada ventana, para la traza
                    })

        # ---- Mantenciones PROGRAMADAS de equipos (Menu 12) ----
        # Se sortean de nuevo en CADA simulacion (duracion y exito pueden ser
        # aleatorios), y se guardan como ventanas de indisponibilidad sobre la
        # instancia especifica del equipo (recurso + instancia_id) a la que apuntan.
        for mantencion in mantenciones_programadas:
            recurso_m = mantencion.get('recurso')
            inst_id_m = mantencion.get('instancia_id')
            if recurso_m not in instancias_recursos:
                continue
            inst_obj = next((i for i in instancias_recursos[recurso_m] if i['id'] == inst_id_m), None)
            if inst_obj is None:
                continue  # la instancia no existe en esta corrida (ej. flota menor a lo configurado)

            ventanas_m, eventos_m = generar_ventanas_mantenimiento_instancia(
                mantencion, duracion_turno, tiempo_limite
            )
            inst_obj['ventanas_mantenimiento'].extend(ventanas_m)
            inst_obj['ventanas_mantenimiento'].sort(key=lambda v: v[0])
            for (v_ini_m, v_fin_m) in ventanas_m:
                inst_obj['mantenciones_info'].append(
                    {'ini': v_ini_m, 'fin': v_fin_m, 'nombre': mantencion['nombre']}
                )

            for ev in eventos_m:
                if ev['Duration'] <= 1e-6:
                    continue
                etiqueta_intento = "" if ev['Intento'] == 1 else f" (reintento N°{ev['Intento']}, mantención anterior falló)"
                traza_eventos.append({
                    'Simulation_ID': sim + 1, 'Cycle': -1,
                    'Actividad': f"Mantención Programada '{mantencion['nombre']}'{etiqueta_intento}"
                                 + ("" if ev['Exito'] else " — FALLÓ, el equipo vuelve a mantención"),
                    'Recurso': f"{recurso_m}_{inst_id_m}",
                    'Detalle_Mantencion': mantencion['nombre'],
                    'type': 'maintenance' if ev['Exito'] else 'maintenance_failed',
                    'Start': round(ev['Start'], 2), 'Finish': round(ev['Finish'], 2),
                    'Duration': round(ev['Duration'], 2),
                    'Duration_Muestreada': round(ev['Duracion_Muestreada'], 4),
                    'Truncado_Horizonte': bool(ev['Duracion_Muestreada'] > ev['Duration'] + 1e-6),
                    'Start_X_Front': None, 'Start_Y_Front': None,
                    'End_X_Front': None, 'End_Y_Front': None,
                    'Cycle_Advance_Length_m': 0, 'Resource_Origin_Tunnel': 'N/A',
                    'Resource_Destination_Tunnel': 'N/A', 'Travel_Time_Actual': 0,
                    'Travel_Speed_Used_m_h': 0, 'Travel_Distance_m': 0,
                    'Es_Personalizada': False
                })

        # ---- Demoras programadas por CAMBIO DE TURNO (Menú 5) ----
        # En cada fin de turno (12 h en 12x2, 8 h en 8x3) todos los equipos se detienen una demora
        # sorteada de la distribución elegida. Se modela como una ventana de indisponibilidad más
        # sobre TODAS las instancias (igual que una mantención: pausa el trabajo en curso y empuja
        # el inicio de lo que cae dentro), por lo que alarga ciclos y avances y llega a todos los
        # resultados vía eventos 'delay_shift_change'.
        vistos_turno_sim = set()
        ventanas_turno_sim = generar_ventanas_cambio_turno(demoras_cambio_turno, duracion_turno, tiempo_limite)
        for (v_ini_t, v_fin_t) in ventanas_turno_sim:
            for _rec_t, _insts_t in instancias_recursos.items():
                for _inst_t in _insts_t:
                    _inst_t['ventanas_mantenimiento'].append((v_ini_t, v_fin_t))
                    _inst_t['mantenciones_info'].append(
                        {'ini': v_ini_t, 'fin': v_fin_t, 'nombre': NOMBRE_CAMBIO_TURNO}
                    )
            traza_eventos.append({
                'Simulation_ID': sim + 1, 'Cycle': -1,
                'Actividad': f"Cambio de turno (fin turno {int(round(v_ini_t / duracion_turno))})",
                'Recurso': 'N/A', 'type': 'shift_change_window',
                'Start': round(v_ini_t, 2), 'Finish': round(v_fin_t, 2),
                'Duration': round(v_fin_t - v_ini_t, 2),
                'Start_X_Front': None, 'Start_Y_Front': None,
                'End_X_Front': None, 'End_Y_Front': None,
                'Cycle_Advance_Length_m': 0, 'Resource_Origin_Tunnel': 'N/A',
                'Resource_Destination_Tunnel': 'N/A', 'Travel_Time_Actual': 0,
                'Travel_Speed_Used_m_h': 0, 'Travel_Distance_m': 0,
                'Es_Personalizada': False
            })
        if ventanas_turno_sim:
            for _rec_t, _insts_t in instancias_recursos.items():
                for _inst_t in _insts_t:
                    _inst_t['ventanas_mantenimiento'].sort(key=lambda v: v[0])

        # Registrar en la traza los cambios de flota que ocurren dentro del horizonte simulado,
        # para que se vean en la carta Gantt / resultados junto con su demora de activacion.
        # Se registran DOS eventos por cambio:
        #  1) 'fleet_change' (como antes): trazabilidad global del cambio de flota, no asociado
        #     a un tunel (se sigue mostrando en el expander de "Linea de tiempo de cambios de flota").
        #  2) 'delay_fleet_change' (NUEVO): solo si demora_horas > 0. Este SI se asocia a cada
        #     frente que usa ese recurso (Resource_Origin_Tunnel = frente), y su 'type' esta
        #     incluido en la lista de eventos de ciclo/Gantt, para que aparezca como una demora
        #     mas ("Demora cambio de recurso {recurso}") en el Resumen de Tiempos y en la Carta
        #     Gantt de cada tunel afectado.
        for cambio in registro_cambios_flota:
            if cambio['tiempo_efectivo'] <= tiempo_limite:
                traza_eventos.append({
                    'Simulation_ID': sim + 1, 'Cycle': -1,
                    'Actividad': f"Cambio de flota: {cambio['tipo']} ({cambio['cantidad_anterior']} -> {cambio['cantidad_nueva']})",
                    'Recurso': cambio['recurso'], 'type': 'fleet_change',
                    'Start': round(cambio['tiempo_efectivo'], 2),
                    'Finish': round(cambio['tiempo_efectivo'], 2),
                    'Duration': round(cambio['demora_horas'], 2),
                    'Start_X_Front': None, 'Start_Y_Front': None,
                    'End_X_Front': None, 'End_Y_Front': None,
                    'Cycle_Advance_Length_m': 0, 'Resource_Origin_Tunnel': 'N/A',
                    'Resource_Destination_Tunnel': 'N/A', 'Travel_Time_Actual': 0,
                    'Travel_Speed_Used_m_h': 0, 'Travel_Distance_m': 0,
                    'Es_Personalizada': True,
                    'Turno_Plan': cambio['turno'], 'Hora_Plan': cambio['hora_inicio'],
                    'Definido_Por': cambio['definido_por'], 'Demora_Aplicada_h': cambio['demora_horas']
                })

                if cambio.get('demora_horas', 0.0) > 1e-9 and cambio['recurso'] in recursos_necesarios:
                    for frente_afectado in frentes_nombres:
                        pos_x_fc, pos_y_fc = calcular_posicion_avanzada(
                            frentes_info[frente_afectado]['xi'], frentes_info[frente_afectado]['yi'],
                            frentes_info[frente_afectado]['xf'], frentes_info[frente_afectado]['yf'],
                            0, frentes_info[frente_afectado]['distancia']
                        )
                        traza_eventos.append({
                            'Simulation_ID': sim + 1, 'Cycle': -1,
                            'Actividad': f"Demora cambio de recurso {cambio['recurso']}",
                            'Recurso': cambio['recurso'], 'type': 'delay_fleet_change',
                            'Start': round(cambio['tiempo_efectivo'], 2),
                            'Finish': round(cambio['tiempo_efectivo'] + cambio['demora_horas'], 2),
                            'Duration': round(cambio['demora_horas'], 2),
                            'Start_X_Front': round(pos_x_fc, 2), 'Start_Y_Front': round(pos_y_fc, 2),
                            'End_X_Front': round(pos_x_fc, 2), 'End_Y_Front': round(pos_y_fc, 2),
                            'Cycle_Advance_Length_m': 0, 'Resource_Origin_Tunnel': frente_afectado,
                            'Resource_Destination_Tunnel': 'N/A', 'Travel_Time_Actual': 0,
                            'Travel_Speed_Used_m_h': 0, 'Travel_Distance_m': 0,
                            'Es_Personalizada': False
                        })

        # Estado de los frentes
        avances_frentes = {frente: 0 for frente in frentes_nombres}
        # Captura del avance REAL acumulado al cierre de cada turno, tomado directamente
        # de avances_frentes (la misma variable que usa "resultados"/Avance Final). Esto
        # reemplaza la reconstruccion posterior desde la traza de eventos (que contaba
        # ciclos con actividad registrada aunque estuvieran truncados por tiempo_limite o
        # por bloqueo de recurso, inflando el avance por turno respecto al Avance Final real).
        avance_por_turno_real = {frente: {} for frente in frentes_nombres}
        siguiente_turno_a_registrar = {frente: 1 for frente in frentes_nombres}
        secuencia_actual_frente = {frente: 0 for frente in frentes_nombres}  # Índice en secuencias_ordenadas
        ciclos_completados_frente = {frente: 0 for frente in frentes_nombres}
        # Frentes bloqueados permanentemente por falta absoluta de un recurso requerido
        # (cantidad 0 ahora y en todo el resto del horizonte de simulación).
        frente_bloqueado = {frente: False for frente in frentes_nombres}
        # Registro de ventanas de BLOQUEO TOTAL (Menu 14) ya escritas en la traza para este
        # 'sim', para no duplicar la misma barra cada vez que un evento cae y se reprograma
        # dentro de la misma ventana.
        ventanas_bloqueo_total_trazadas = {frente: set() for frente in frentes_nombres}
        
        # Cola de eventos: (tiempo, tipo, datos)
        event_queue = []
        event_counter = 0  # Para desempatar eventos con mismo tiempo
        
        # Índices para velocidades
        idx_velocidades = {clave_v: sim * 100 for clave_v in velocidades_flat.keys()}
        
        global_cycle_count = 0
        
        # Inicializar: agregar eventos para iniciar primera secuencia de cada frente
        for frente in frentes_nombres:
            heapq.heappush(event_queue, (0, event_counter, 'start_sequence', {
                'frente': frente,
                'secuencia_idx': 0
            }))
            event_counter += 1
        
        # Procesar eventos
        while event_queue:
            current_time, _, event_type, event_data = heapq.heappop(event_queue)
            
            if current_time >= tiempo_limite:
                break

            # Registrar avance REAL acumulado (avances_frentes) en cada turno cuyo limite
            # de tiempo ya quedo atras respecto al reloj de la simulacion. Se hace para
            # todos los frentes (no solo el del evento actual) para que un frente sin mas
            # eventos propios (bloqueado, terminado, sin recurso) siga registrando su
            # avance congelado en los turnos que van pasando.
            for frente_reg in frentes_nombres:
                turno_pendiente = siguiente_turno_a_registrar[frente_reg]
                while turno_pendiente * duracion_turno <= current_time:
                    avance_por_turno_real[frente_reg][turno_pendiente] = min(
                        avances_frentes[frente_reg], frentes_info[frente_reg]['distancia']
                    )
                    turno_pendiente += 1
                siguiente_turno_a_registrar[frente_reg] = turno_pendiente

            frente = event_data['frente']
            
            # Verificar si el frente ya completó su distancia
            if avances_frentes[frente] >= frentes_info[frente]['distancia']:
                continue

            # Verificar si el frente quedó bloqueado permanentemente por falta de un recurso
            if frente_bloqueado[frente]:
                continue

            # Verificar BLOQUEO TOTAL de túnel (Menu 14): mientras current_time caiga dentro
            # de una ventana bloqueada, el frente no ejecuta NINGÚN evento (avance 0, no se le
            # asigna ningún recurso) — se reprograma el mismo evento justo al finalizar la
            # ventana, sin tocar secuencia_actual_frente ni avances_frentes, para que retome
            # exactamente donde iba una vez terminado el bloqueo. Cubre cualquier tipo de
            # evento porque en este motor todos pasan por 'start_sequence'.
            fin_bloqueo_total = frente_en_bloqueo_total(frente, current_time, bloqueos_totales_frente)
            if fin_bloqueo_total is not None:
                clave_ventana = round(fin_bloqueo_total, 6)
                if clave_ventana not in ventanas_bloqueo_total_trazadas[frente]:
                    ventanas_bloqueo_total_trazadas[frente].add(clave_ventana)
                    pos_x_bt, pos_y_bt = calcular_posicion_avanzada(
                        frentes_info[frente]['xi'], frentes_info[frente]['yi'],
                        frentes_info[frente]['xf'], frentes_info[frente]['yf'],
                        avances_frentes[frente], frentes_info[frente]['distancia']
                    )
                    traza_eventos.append({
                        'Simulation_ID': sim + 1, 'Cycle': global_cycle_count, 'Ciclo_Frente': ciclos_completados_frente[frente] + 1, 'Secuencia': 0,
                        'Actividad': 'BLOQUEO TOTAL DE TÚNEL (Restricción Física)',
                        'Recurso': 'N/A', 'type': 'delay_bloqueo_total',
                        'Start': round(current_time, 2), 'Finish': round(fin_bloqueo_total, 2),
                        'Duration': round(fin_bloqueo_total - current_time, 2),
                        'Start_X_Front': round(pos_x_bt, 2), 'Start_Y_Front': round(pos_y_bt, 2),
                        'End_X_Front': round(pos_x_bt, 2), 'End_Y_Front': round(pos_y_bt, 2),
                        'Cycle_Advance_Length_m': 0, 'Resource_Origin_Tunnel': frente,
                        'Resource_Destination_Tunnel': 'N/A', 'Travel_Time_Actual': 0,
                        'Travel_Speed_Used_m_h': 0, 'Travel_Distance_m': 0,
                        'Es_Personalizada': False
                    })
                heapq.heappush(event_queue, (fin_bloqueo_total, event_counter, event_type, event_data))
                event_counter += 1
                continue
            
            if event_type == 'start_sequence':
                secuencia_idx = event_data['secuencia_idx']
                
                # Verificar si todavía hay secuencias por ejecutar
                if secuencia_idx >= len(secuencias_ordenadas):
                    # Ciclo completo, avanzar y reiniciar secuencias
                    
                    metros_avance_ciclo = metros_ciclo(frente)

                    avances_frentes[frente] += metros_avance_ciclo
                    ciclos_completados_frente[frente] += 1
                    
                    # Si no completó la distancia, iniciar nuevo ciclo
                    if avances_frentes[frente] < frentes_info[frente]['distancia']:
                        heapq.heappush(event_queue, (current_time, event_counter, 'start_sequence', {
                            'frente': frente,
                            'secuencia_idx': 0
                        }))
                        event_counter += 1
                    continue
                
                secuencia_num = secuencias_ordenadas[secuencia_idx]
                # Ya viene precalculado antes del loop de Monte Carlo (acts_por_secuencia_frente,
                # ver mas arriba): es una lista de dicts, no un DataFrame -- antes esto se
                # recalculaba aca mismo con un DataFrame.apply(axis=1), en cada evento, para el
                # mismo resultado deterministico repetido miles de veces por corrida.
                acts_secuencia = acts_por_secuencia_frente[(secuencia_num, frente)]

                if not acts_secuencia:
                    # Si no hay actividades para esta secuencia/frente, pasar a la siguiente secuencia inmediatamente.
                    heapq.heappush(event_queue, (current_time, event_counter, 'start_sequence', {
                        'frente': frente,
                        'secuencia_idx': secuencia_idx + 1
                    }))
                    event_counter += 1
                    continue

                # Aplicar demoras al inicio de cada ciclo (solo en secuencia 0)
                tiempo_inicio_secuencia = current_time
                if secuencia_idx == 0:
                    global_cycle_count += 1
                    
                    # Demoras constantes (FH/OC)
                    demora_fh_oc = demoras_constantes.get(frente, 0)
                    if demora_fh_oc > 0:
                        pos_x, pos_y = calcular_posicion_avanzada(
                            frentes_info[frente]['xi'], frentes_info[frente]['yi'],
                            frentes_info[frente]['xf'], frentes_info[frente]['yf'],
                            avances_frentes[frente], frentes_info[frente]['distancia']
                        )
                        
                        traza_eventos.append({
                            'Simulation_ID': sim + 1, 'Cycle': global_cycle_count, 'Ciclo_Frente': ciclos_completados_frente[frente] + 1, 'Secuencia': 0,
                            'Actividad': 'Demora por Frente Hundimiento, Restricción por Obras Civiles',
                            'Recurso': 'N/A', 'type': 'delay_fh_oc',
                            'Start': round(tiempo_inicio_secuencia, 2),
                            'Finish': round(tiempo_inicio_secuencia + demora_fh_oc, 2),
                            'Duration': round(demora_fh_oc, 2),
                            'Start_X_Front': round(pos_x, 2), 'Start_Y_Front': round(pos_y, 2),
                            'End_X_Front': round(pos_x, 2), 'End_Y_Front': round(pos_y, 2),
                            'Cycle_Advance_Length_m': 0, 'Resource_Origin_Tunnel': frente,
                            'Resource_Destination_Tunnel': 'N/A', 'Travel_Time_Actual': 0,
                            'Travel_Speed_Used_m_h': 0, 'Travel_Distance_m': 0,
                            'Es_Personalizada': False # NO es personalizada
                        })
                        tiempo_inicio_secuencia += demora_fh_oc
                    
                    # Demoras por turno (R) — turno 1-based: turno 1 = [0, duracion_turno)
                    turno_inicio = int(tiempo_inicio_secuencia / duracion_turno) + 1
                    for res in demoras_por_turno:
                        if res['Frente'] == frente and res['Turno'] == turno_inicio:
                            demora_r = res['Demora']
                            pos_x, pos_y = calcular_posicion_avanzada(
                                frentes_info[frente]['xi'], frentes_info[frente]['yi'],
                                frentes_info[frente]['xf'], frentes_info[frente]['yf'],
                                avances_frentes[frente], frentes_info[frente]['distancia']
                            )
                            traza_eventos.append({
                                'Simulation_ID': sim + 1, 'Cycle': global_cycle_count, 'Ciclo_Frente': ciclos_completados_frente[frente] + 1, 'Secuencia': 0,
                                # OJO: antes esto guardaba solo res['Tipo'] (ej. "Polvorazo"), sin
                                # el prefijo "Demora...". La Carta Gantt (calcular_datos_gantt) y
                                # el resto de las tablas que agrupan por 'Actividad' usan justamente
                                # ese prefijo para pintar la barra de rojo y ordenarla primero como
                                # demora; sin el, la barra existia pero se mezclaba sin distinguirse
                                # entre las actividades normales (color azul, orden alfabetico), y
                                # el usuario no la notaba en el grafico.
                                'Actividad': f"Demora por Restricción ({res['Tipo']})",
                                'Recurso': 'N/A', 'type': 'delay_res',
                                'Start': round(tiempo_inicio_secuencia, 2),
                                'Finish': round(tiempo_inicio_secuencia + demora_r, 2),
                                'Duration': round(demora_r, 2),
                                'Start_X_Front': round(pos_x, 2), 'Start_Y_Front': round(pos_y, 2),
                                'End_X_Front': round(pos_x, 2), 'End_Y_Front': round(pos_y, 2),
                                'Cycle_Advance_Length_m': 0, 'Resource_Origin_Tunnel': frente,
                                'Resource_Destination_Tunnel': 'N/A', 'Travel_Time_Actual': 0,
                                'Travel_Speed_Used_m_h': 0, 'Travel_Distance_m': 0,
                                'Es_Personalizada': False # NO es personalizada
                            })
                            tiempo_inicio_secuencia += demora_r
                    
                    # --- NUEVO: Demoras por Falla de Equipo ---
                    if frente in mapa_fallas_equipos:
                        falla_rule = mapa_fallas_equipos[frente]
                        demora_total_falla = 0
                        fallas_ocurridas = []  # [(recurso, demora), ...] de ESTE ciclo

                        for recurso, params in falla_rule['fallas_por_recurso'].items():
                            prob_falla = params.get('probabilidad', 0)

                            if prob_falla > 0 and np.random.rand() < (prob_falla / 100.0):
                                # La demora puede ser fija (Manual) o muestreada de una
                                # distribución estadística propia por recurso (Menú 11),
                                # con el mismo patrón de vista previa/filtro del resto de la app.
                                if params.get('modo_demora') == 'Distribución estadística':
                                    demora_falla = float(generar_tiempos_batch(
                                        params['params_distribucion'], params['distribucion'], 1
                                    )[0])
                                else:
                                    demora_falla = float(params.get('demora', 0) or 0)
                                demora_falla = max(0.0, demora_falla)
                                demora_total_falla += demora_falla
                                fallas_ocurridas.append((recurso, demora_falla))

                        if demora_total_falla > 0:
                            pos_x, pos_y = calcular_posicion_avanzada(
                                frentes_info[frente]['xi'], frentes_info[frente]['yi'],
                                frentes_info[frente]['xf'], frentes_info[frente]['yf'],
                                avances_frentes[frente], frentes_info[frente]['distancia']
                            )

                            # Se registra UN evento de traza POR CADA recurso que falló en este
                            # ciclo (en vez de uno agregado), para poder reportar en resultados
                            # (Menú 20) cuántas veces y cuánta demora aportó cada equipo, igual
                            # espíritu que el Menú 19 para mantenciones programadas.
                            t_cursor_falla = tiempo_inicio_secuencia
                            for recurso_fallado, demora_r_falla in fallas_ocurridas:
                                traza_eventos.append({
                                    'Simulation_ID': sim + 1, 'Cycle': global_cycle_count, 'Ciclo_Frente': ciclos_completados_frente[frente] + 1, 'Secuencia': 0,
                                    'Actividad': f'Demora por Falla de Equipo ({recurso_fallado})',
                                    'Recurso': recurso_fallado,
                                    'type': 'delay_equipment_failure',
                                    'Start': round(t_cursor_falla, 2),
                                    'Finish': round(t_cursor_falla + demora_r_falla, 2),
                                    'Duration': round(demora_r_falla, 2),
                                    'Start_X_Front': round(pos_x, 2), 'Start_Y_Front': round(pos_y, 2),
                                    'End_X_Front': round(pos_x, 2), 'End_Y_Front': round(pos_y, 2),
                                    'Cycle_Advance_Length_m': 0, 'Resource_Origin_Tunnel': frente,
                                    'Resource_Destination_Tunnel': 'N/A', 'Travel_Time_Actual': 0,
                                    'Travel_Speed_Used_m_h': 0, 'Travel_Distance_m': 0,
                                    'Es_Personalizada': False
                                })
                                t_cursor_falla += demora_r_falla
                            tiempo_inicio_secuencia += demora_total_falla
                
                # Procesar cada actividad de la secuencia
                tiempo_max_secuencia = tiempo_inicio_secuencia
                recursos_asignados_secuencia = []  # Permite mas de una actividad con el mismo tipo de recurso.
                recursos_reservados_secuencia = set()
                bloqueo_permanente = None  # Si se detecta falta absoluta de un recurso requerido
                
                # PASO 1: Asignar recursos a actividades (sin registrar viajes aún)
                for act in acts_secuencia:
                    recurso = act['Recurso']
                    
                    # Si no requiere recurso, ejecutar directamente
                    if pd.isna(recurso):
                        duracion_act = generar_tiempos_batch(act['Tiempo'], act['Distribucion'], 1)[0]
                        _geo_cols = {}
                        if _mwd_rt is not None:
                            duracion_act, _geo_cols = mwd_muestrear(
                                _mwd_rt, sim, frente, act, ciclos_completados_frente[frente] + 1, duracion_act, avances_frentes[frente],
                                metros_ciclo(frente))
                        # Duracion MUESTREADA completa (la que sale de la distribucion, ya
                        # truncada al rango elegido). 'Duration' puede quedar mas corta si el
                        # horizonte de simulacion corta la actividad; los analisis de la
                        # distribucion (Monte Carlo) usan esta columna, no 'Duration'.
                        duracion_muestreada = float(duracion_act)
                        
                        start_act = tiempo_inicio_secuencia
                        finish_act = start_act + duracion_act
                        truncado_horizonte = False
                        
                        if finish_act > tiempo_limite:
                            finish_act = tiempo_limite
                            duracion_act = finish_act - start_act
                            truncado_horizonte = True
                        
                        if duracion_act > 1e-6:
                            pos_x, pos_y = calcular_posicion_avanzada(
                                frentes_info[frente]['xi'], frentes_info[frente]['yi'],
                                frentes_info[frente]['xf'], frentes_info[frente]['yf'],
                                avances_frentes[frente], frentes_info[frente]['distancia']
                            )
                            
                            traza_eventos.append({
                                'Simulation_ID': sim + 1, 'Cycle': global_cycle_count, 'Ciclo_Frente': ciclos_completados_frente[frente] + 1, 'Secuencia': secuencia_num,
                                'Actividad': act['Actividad'], 'Recurso': 'N/A', 'type': 'activity',
                                **_geo_cols,
                                'Start': round(start_act, 2), 'Finish': round(finish_act, 2),
                                'Duration': round(duracion_act, 2),
                                'Duration_Muestreada': round(duracion_muestreada, 4),
                                'Truncado_Horizonte': truncado_horizonte,
                                'Start_X_Front': round(pos_x, 2), 'Start_Y_Front': round(pos_y, 2),
                                'End_X_Front': round(pos_x, 2), 'End_Y_Front': round(pos_y, 2),
                                'Cycle_Advance_Length_m': metros_ciclo(frente), 'Resource_Origin_Tunnel': frente,
                                'Resource_Destination_Tunnel': frente, 'Travel_Time_Actual': 0,
                                'Travel_Speed_Used_m_h': 0, 'Travel_Distance_m': 0,
                                'Es_Personalizada': act.get('Es_Personalizada', False) # <--- PROPAGAR FLAG
                            })
                        
                        tiempo_max_secuencia = max(tiempo_max_secuencia, finish_act)
                        continue
                    
                    # Buscar recurso disponible. Si el recurso ni siquiera existe en la
                    # configuracion (cantidad 0 y nunca lo aumenta el plan de flota), esta
                    # actividad JAMAS podra ejecutarse: el frente queda bloqueado.
                    if recurso not in instancias_recursos:
                        bloqueo_permanente = recurso
                        break
                    
                    mejor_inst = None
                    menor_tiempo_llegada = float('inf')
                    info_viaje = None
                    
                    pos_x_frente, pos_y_frente = calcular_posicion_avanzada(
                        frentes_info[frente]['xi'], frentes_info[frente]['yi'],
                        frentes_info[frente]['xf'], frentes_info[frente]['yf'],
                        avances_frentes[frente], frentes_info[frente]['distancia']
                    )

                    for inst in instancias_recursos[recurso]:
                        if (recurso, inst['id']) in recursos_reservados_secuencia:
                            continue
                        # Una instancia ya retirada definitivamente para este instante nunca
                        # podra atender esta actividad.
                        if tiempo_inicio_secuencia >= inst.get('activa_hasta', float('inf')):
                            continue
                        tiempo_viaje = 0
                        distancia_viaje = 0
                        velocidad_km_h = 15.0
                        # Velocidad propia de esta unidad (si se definio por unidad) o la del recurso.
                        clave_vel = (recurso, inst['id']) if (recurso, inst['id']) in velocidades_precalculadas else recurso
                        necesita_viaje = inst['frente_actual'] != frente
                        
                        if necesita_viaje:
                            # --- NUEVA LÓGICA DE VIAJE EN U ---
                            frente_origen = inst['frente_actual']
                            
                            # 1. Salida del túnel actual (desde posición actual hasta entrada del túnel origen)
                            entrada_origen_x = frentes_info[frente_origen]['xi']
                            entrada_origen_y = frentes_info[frente_origen]['yi']
                            distancia_salida = calcular_distancia(inst['x'], inst['y'], entrada_origen_x, entrada_origen_y)
                            
                            # 2. Viaje entre entradas de túneles (entrada origen a entrada destino)
                            entrada_destino_x = frentes_info[frente]['xi']
                            entrada_destino_y = frentes_info[frente]['yi']
                            distancia_entre_tuneles = calcular_distancia(entrada_origen_x, entrada_origen_y, entrada_destino_x, entrada_destino_y)
                            
                            # 3. Entrada al túnel destino (desde entrada hasta posición del frente)
                            distancia_entrada = calcular_distancia(entrada_destino_x, entrada_destino_y, pos_x_frente, pos_y_frente)
                            
                            # Distancia total del viaje en U
                            distancia_viaje = distancia_salida + distancia_entre_tuneles + distancia_entrada
                            
                            # Solo calcular viaje si la distancia es significativa
                            if distancia_viaje > 0.1:  # Más de 10cm
                                idx_v = idx_velocidades.get(clave_vel, 0)
                                if clave_vel in velocidades_precalculadas and idx_v < len(velocidades_precalculadas[clave_vel]):
                                    velocidad_km_h = velocidades_precalculadas[clave_vel][idx_v]
                                else:
                                    velocidad_km_h = np.mean(velocidades_precalculadas.get(clave_vel, [15.0]))
                                
                                distancia_km = distancia_viaje / 1000.0
                                tiempo_viaje = distancia_km / velocidad_km_h if velocidad_km_h > 0 else 0
                            else:
                                necesita_viaje = False  # Distancia insignificante
                            # --- FIN NUEVA LÓGICA ---
                        
                        # Instante mas temprano en que esta instancia podria efectivamente
                        # atender la actividad: lo maximo entre (a) cuando queda libre de su
                        # tarea anterior, (b) el inicio de esta secuencia, y (c) el instante en
                        # que la instancia entra en vigencia segun el plan de flota (si aun no
                        # se ha incorporado, hay que ESPERAR a que se incorpore, no descartarla).
                        tiempo_disponible_inst = max(inst['disponible_en'], tiempo_inicio_secuencia, inst.get('activa_desde', 0.0))
                        # Si la instancia tiene mantención programada (Menu 12) y ese instante
                        # cae dentro de una ventana de mantención, se empuja al final de esta
                        # (y de cualquier otra que venga encadenada, ej. por reintento fallido).
                        tiempo_disponible_inst = _tiempo_disponible_tras_mantenciones(
                            tiempo_disponible_inst, inst.get('ventanas_mantenimiento')
                        )
                        # Si para cuando la instancia estaria libre y vigente ya fue retirada, no sirve.
                        if tiempo_disponible_inst >= inst.get('activa_hasta', float('inf')):
                            continue

                        tiempo_llegada = tiempo_disponible_inst + tiempo_viaje
                        if tiempo_llegada >= inst.get('activa_hasta', float('inf')):
                            continue
                        
                        if tiempo_llegada < menor_tiempo_llegada:
                            menor_tiempo_llegada = tiempo_llegada
                            mejor_inst = inst
                            info_viaje = {
                                'necesita_viaje': necesita_viaje,
                                'distancia': distancia_viaje,
                                'tiempo': tiempo_viaje,
                                'velocidad': velocidad_km_h,
                                'idx_velocidad': idx_velocidades.get(clave_vel, 0),
                                'clave_vel': clave_vel
                            }

                    if mejor_inst is None:
                        # Ninguna instancia de este recurso podra atender la actividad, ni ahora
                        # ni en el resto del horizonte simulado (todas retiradas / inexistentes
                        # para siempre desde este instante). El frente queda bloqueado: no puede
                        # completar el ciclo sin este recurso.
                        bloqueo_permanente = recurso
                        break
                    
                    # Guardar la asignación para procesarla después
                    recursos_reservados_secuencia.add((recurso, mejor_inst['id']))
                    recursos_asignados_secuencia.append({
                        'recurso': recurso,
                        'instancia': mejor_inst,
                        'info_viaje': info_viaje,
                        'actividad': act,
                        'pos_frente': (pos_x_frente, pos_y_frente)
                    })

                # Si se detectó falta absoluta de un recurso requerido, el frente no puede
                # completar este ciclo: se bloquea su avance de forma permanente a partir de
                # este instante (no se procesan las asignaciones parciales ni se agenda la
                # siguiente secuencia).
                if bloqueo_permanente is not None:
                    frente_bloqueado[frente] = True
                    pos_x_b, pos_y_b = calcular_posicion_avanzada(
                        frentes_info[frente]['xi'], frentes_info[frente]['yi'],
                        frentes_info[frente]['xf'], frentes_info[frente]['yf'],
                        avances_frentes[frente], frentes_info[frente]['distancia']
                    )
                    traza_eventos.append({
                        'Simulation_ID': sim + 1, 'Cycle': global_cycle_count, 'Ciclo_Frente': ciclos_completados_frente[frente] + 1, 'Secuencia': secuencia_num,
                        'Actividad': f"BLOQUEO: sin '{bloqueo_permanente}' disponible para completar el ciclo",
                        'Recurso': bloqueo_permanente, 'type': 'blocked_no_resource',
                        'Start': round(tiempo_inicio_secuencia, 2), 'Finish': round(tiempo_limite, 2),
                        'Duration': round(tiempo_limite - tiempo_inicio_secuencia, 2),
                        'Start_X_Front': round(pos_x_b, 2), 'Start_Y_Front': round(pos_y_b, 2),
                        'End_X_Front': round(pos_x_b, 2), 'End_Y_Front': round(pos_y_b, 2),
                        'Cycle_Advance_Length_m': 0, 'Resource_Origin_Tunnel': frente,
                        'Resource_Destination_Tunnel': 'N/A', 'Travel_Time_Actual': 0,
                        'Travel_Speed_Used_m_h': 0, 'Travel_Distance_m': 0,
                        'Es_Personalizada': False
                    })
                    continue
                
                # PASO 2: Procesar recursos asignados (registrar viajes y actividades)
                for asignacion in recursos_asignados_secuencia:
                    recurso = asignacion['recurso']
                    mejor_inst = asignacion['instancia']
                    info_viaje = asignacion['info_viaje']
                    act = asignacion['actividad']
                    pos_x_frente, pos_y_frente = asignacion['pos_frente']
                    
                    # Tiempo inicial del recurso
                    tiempo_disponible_recurso = max(
                        mejor_inst['disponible_en'], tiempo_inicio_secuencia, mejor_inst.get('activa_desde', 0.0)
                    )
                    # Mantencion programada (Menu 12): si la instancia esta en mantencion en ese
                    # instante, ni viaja ni trabaja hasta que termine (igual que al elegirla en PASO 1).
                    _t_antes_mant = tiempo_disponible_recurso
                    tiempo_disponible_recurso = _tiempo_disponible_tras_mantenciones(
                        tiempo_disponible_recurso, mejor_inst.get('ventanas_mantenimiento')
                    )
                    if tiempo_disponible_recurso > _t_antes_mant + 1e-6:
                        traza_eventos.extend(_eventos_demora_mantencion(
                            sim, global_cycle_count, ciclos_completados_frente[frente] + 1, secuencia_num,
                            frente, recurso, mejor_inst, [(_t_antes_mant, tiempo_disponible_recurso)],
                            tiempo_limite, pos_x_frente, pos_y_frente, vistos_turno_sim
                        ))
                    
                    # Registrar viaje si es necesario
                    if info_viaje['necesita_viaje'] and info_viaje['tiempo'] > 1e-3:
                        start_viaje = tiempo_disponible_recurso
                        finish_viaje = start_viaje + info_viaje['tiempo']
                        
                        if finish_viaje > tiempo_limite:
                            finish_viaje = tiempo_limite
                        
                        if finish_viaje > start_viaje + 1e-6:
                            traza_eventos.append({
                                'Simulation_ID': sim + 1, 'Cycle': global_cycle_count, 'Ciclo_Frente': ciclos_completados_frente[frente] + 1, 'Secuencia': secuencia_num,
                                'Actividad': act['Actividad'], 
                                'Recurso': f"{recurso}_{mejor_inst['id']}", 
                                'type': 'travel',
                                'Start': round(start_viaje, 2), 'Finish': round(finish_viaje, 2),
                                'Duration': round(finish_viaje - start_viaje, 4),
                                'Start_X_Front': round(mejor_inst['x'], 2),
                                'Start_Y_Front': round(mejor_inst['y'], 2),
                                'End_X_Front': round(pos_x_frente, 2),
                                'End_Y_Front': round(pos_y_frente, 2),
                                'Cycle_Advance_Length_m': 0,
                                'Resource_Origin_Tunnel': mejor_inst['frente_actual'],
                                'Resource_Destination_Tunnel': frente,
                                'Travel_Time_Actual': round(info_viaje['tiempo'], 4),
                                'Travel_Speed_Used_m_h': round(info_viaje['velocidad'] * 1000, 2),
                                'Travel_Distance_m': round(info_viaje['distancia'], 2),
                                'Es_Personalizada': False # Los viajes no son una actividad personalizada para el Gantt
                            })
                            
                            # Actualizar posición después del viaje
                            mejor_inst['x'] = pos_x_frente
                            mejor_inst['y'] = pos_y_frente
                            mejor_inst['frente_actual'] = frente
                            mejor_inst['disponible_en'] = finish_viaje
                            tiempo_disponible_recurso = finish_viaje
                            
                            idx_velocidades[info_viaje['clave_vel']] = info_viaje['idx_velocidad'] + 1
                    
                    # Ejecutar actividad
                    duracion_act = generar_tiempos_batch(act['Tiempo'], act['Distribucion'], 1)[0]
                    _geo_cols = {}
                    if _mwd_rt is not None:
                        duracion_act, _geo_cols = mwd_muestrear(
                            _mwd_rt, sim, frente, act, ciclos_completados_frente[frente] + 1, duracion_act, avances_frentes[frente],
                                metros_ciclo(frente))

                    # --- MODIFICACION: Aplicar ponderador de recurso por restricción geológica ---
                    if frente in mapa_restriccion_geologica:
                        res_geo = mapa_restriccion_geologica[frente]
                        ponderadores = res_geo.get('ponderadores_recursos', {})
                        if recurso in ponderadores:
                            ponderador_pct = ponderadores[recurso]
                            factor_ajuste = 1 + (ponderador_pct / 100.0)
                            duracion_act *= factor_ajuste
                    # --- FIN MODIFICACION ---

                    # Duracion completa de la actividad (muestreada + ponderador geologico, si
                    # existe), ANTES de que el horizonte de simulacion la pueda cortar.
                    duracion_muestreada = float(duracion_act)
                    # Si el viaje terminó dentro de una ventana de mantención, la actividad espera.
                    start_act = _tiempo_disponible_tras_mantenciones(
                        tiempo_disponible_recurso, mejor_inst.get('ventanas_mantenimiento')
                    )
                    _intervalos_mant = []
                    if start_act > tiempo_disponible_recurso + 1e-6:
                        _intervalos_mant.append((tiempo_disponible_recurso, start_act))
                    # Si una mantención cae a mitad de la actividad, el trabajo se pausa y se reanuda;
                    # cada pausa queda registrada como demora por mantención (trazabilidad).
                    _intervalos_mant.extend(_pausas_por_mantencion(
                        start_act, duracion_act, mejor_inst.get('ventanas_mantenimiento')
                    ))
                    if _intervalos_mant:
                        traza_eventos.extend(_eventos_demora_mantencion(
                            sim, global_cycle_count, ciclos_completados_frente[frente] + 1, secuencia_num,
                            frente, recurso, mejor_inst, _intervalos_mant,
                            tiempo_limite, pos_x_frente, pos_y_frente, vistos_turno_sim
                        ))
                    finish_act = _fin_con_mantenciones(
                        start_act, duracion_act, mejor_inst.get('ventanas_mantenimiento')
                    )
                    duracion_act = finish_act - start_act   # tiempo real transcurrido (incluye la pausa)
                    truncado_horizonte = False
                    
                    if finish_act > tiempo_limite:
                        finish_act = tiempo_limite
                        duracion_act = finish_act - start_act
                        truncado_horizonte = True
                    
                    if duracion_act > 1e-6:
                        traza_eventos.append({
                            'Simulation_ID': sim + 1, 'Cycle': global_cycle_count, 'Ciclo_Frente': ciclos_completados_frente[frente] + 1, 'Secuencia': secuencia_num,
                            'Actividad': act['Actividad'],
                            'Recurso': f"{recurso}_{mejor_inst['id']}", 
                            'type': 'activity',
                            **_geo_cols,
                            'Start': round(start_act, 2), 'Finish': round(finish_act, 2),
                            'Duration': round(duracion_act, 2),
                            'Duration_Muestreada': round(duracion_muestreada, 4),
                            'Truncado_Horizonte': truncado_horizonte,
                            'Start_X_Front': round(pos_x_frente, 2),
                            'Start_Y_Front': round(pos_y_frente, 2),
                            'End_X_Front': round(pos_x_frente, 2),
                            'End_Y_Front': round(pos_y_frente, 2),
                            'Cycle_Advance_Length_m': metros_ciclo(frente),
                            'Resource_Origin_Tunnel': frente,
                            'Resource_Destination_Tunnel': frente,
                            'Travel_Time_Actual': 0, 'Travel_Speed_Used_m_h': 0,
                            'Travel_Distance_m': 0,
                            'Es_Personalizada': act.get('Es_Personalizada', False) # <--- PROPAGAR FLAG
                        })
                    
                    # LIBERACIÓN INMEDIATA del recurso después de la actividad
                    mejor_inst['disponible_en'] = finish_act
                    mejor_inst['x'] = pos_x_frente
                    mejor_inst['y'] = pos_y_frente
                    mejor_inst['frente_actual'] = frente
                    
                    tiempo_max_secuencia = max(tiempo_max_secuencia, finish_act)
                
                # Agregar evento para siguiente secuencia
                secuencia_siguiente = secuencia_idx + 1
                if secuencia_siguiente >= len(secuencias_ordenadas) and tiempo_max_secuencia >= tiempo_limite:
                    # Esta era la ultima secuencia del ciclo y ya no queda tiempo para que el
                    # evento 'start_sequence' que cerraria el ciclo llegue a procesarse (el
                    # bucle principal corta con current_time >= tiempo_limite). Sin este ajuste,
                    # las actividades del ciclo ya quedaban en la traza (avance_actividad) pero
                    # avances_frentes nunca sumaba esos metros: se cerraba el ciclo aqui mismo,
                    # en el mismo instante en que se sabe que no quedan mas secuencias, en vez
                    # de depender de un evento futuro que nunca se alcanza a procesar.
                    metros_avance_ciclo = metros_ciclo(frente)
                    avances_frentes[frente] += metros_avance_ciclo
                    ciclos_completados_frente[frente] += 1
                else:
                    heapq.heappush(event_queue, (tiempo_max_secuencia, event_counter, 'start_sequence', {
                        'frente': frente,
                        'secuencia_idx': secuencia_siguiente
                    }))
                    event_counter += 1
        
        # Guardar resultados
        for frente in frentes_nombres:
            avance_final_frente = min(avances_frentes[frente], frentes_info[frente]['distancia'])
            resultados[frente].append(avance_final_frente)

            # Completar los turnos que quedaron sin registrar (la simulacion termino antes
            # de tiempo_limite, o el event_queue se vacio) con el avance final ya conocido,
            # para que la serie por turno tenga el mismo largo para todos los frentes.
            n_turnos_frente = math.ceil(tiempo_limite / duracion_turno)
            turno_pendiente = siguiente_turno_a_registrar[frente]
            while turno_pendiente <= n_turnos_frente:
                avance_por_turno_real[frente][turno_pendiente] = avance_final_frente
                turno_pendiente += 1

            # Volcar al dict que se retorna/expone (lista ordenada 1..N, alineada 1 a 1
            # con la simulacion "sim+1", igual que resultados[frente]).
            for turno_idx in range(1, n_turnos_frente + 1):
                avance_por_turno_resultado[frente][turno_idx].append(
                    avance_por_turno_real[frente].get(turno_idx, avance_final_frente)
                )

        # Notificar progreso (simulaciones completadas / total) para que la UI pueda
        # mostrar un contador tipo "3/10" y mover la barra en vivo en vez de quedar
        # estatica hasta el final de todo el loop.
        if progress_callback and ((sim + 1) % intervalo_notificacion_progreso == 0 or (sim + 1) == n_simulaciones):
            progress_callback(sim + 1, n_simulaciones)
    
    # Calcular estadísticas de recursos
    estadisticas_recursos = {}
    if traza_eventos:
        df_traza = pd.DataFrame(traza_eventos)
            
        for recurso_tipo in recursos_config.keys():
            if recursos_config[recurso_tipo]['cantidad'] > 0:
                utilizaciones = []
                tiempos_trabajo = []
                tiempos_viaje = []

                # Tiempo disponible real integrando las ventanas de actividad del plan de
                # flota por turno (si no hay cambios de plan, equivale a cantidad_base * tiempo_limite)
                ventanas_recurso = ventanas_por_recurso.get(recurso_tipo, [])
                if ventanas_recurso:
                    tiempo_total_disponible = sum(
                        max(0.0, min(v['activa_hasta'], tiempo_limite) - min(v['activa_desde'], tiempo_limite))
                        for v in ventanas_recurso
                    )
                else:
                    tiempo_total_disponible = tiempo_limite * recursos_config[recurso_tipo]['cantidad']

                for sim_id in range(1, n_simulaciones + 1):
                    df_sim = df_traza[df_traza['Simulation_ID'] == sim_id]
                    
                    tiempo_trabajo_sim = df_sim[(df_sim['Recurso'].str.startswith(recurso_tipo)) & (df_sim['type'] == 'activity')]['Duration'].sum()
                    tiempo_viaje_sim = df_sim[(df_sim['Recurso'].str.startswith(recurso_tipo)) & (df_sim['type'] == 'travel')]['Duration'].sum()
                    
                    tiempo_total_ocupado = tiempo_trabajo_sim + tiempo_viaje_sim
                    
                    utilizacion = (tiempo_total_ocupado / tiempo_total_disponible) * 100 if tiempo_total_disponible > 0 else 0
                    
                    utilizaciones.append(min(100.0, utilizacion))
                    tiempos_trabajo.append(tiempo_trabajo_sim)
                    tiempos_viaje.append(tiempo_viaje_sim)

                estadisticas_recursos[recurso_tipo] = {
                    'utilizacion': utilizaciones, 
                    'tiempo_trabajando': tiempos_trabajo, 
                    'tiempo_viaje': tiempos_viaje
                }

    return resultados, estadisticas_recursos, traza_eventos, ventanas_por_recurso, registro_cambios_flota, avance_por_turno_resultado

# Funciones Auxiliares para el Cálculo por Turnos
@st.cache_data(show_spinner=False)
def calcular_avance_por_turno(_run_id, df_traza_completa, tiempo_limite, sistema_turnos, frentes_info, metros_avance):
    """
    Cacheada con @st.cache_data: agrupa/ordena la traza completa por ciclo y turno, algo
    costoso con cientos de miles de filas. Sin cache, Streamlit repetia este groupby en
    cada rerun (ej. al cambiar el tunel seleccionado en otra seccion), aunque la traza no
    hubiera cambiado. El parametro _run_id (con guion bajo, para que Streamlit no intente
    hashear el DataFrame completo) identifica de forma unica cada corrida de simulacion:
    mientras no cambie, se reutiliza el resultado ya calculado.
    """
    return _calcular_avance_por_turno_impl(df_traza_completa, tiempo_limite, sistema_turnos, frentes_info, metros_avance)


def _calcular_avance_por_turno_impl(df_traza_completa, tiempo_limite, sistema_turnos, frentes_info, metros_avance):
    """Calcula la distancia avanzada por turno para cada simulación y frente."""
    
    if df_traza_completa.empty:
        return {}

    if '12x2' in sistema_turnos:
        duracion_turno = 12.0
    elif '8x3' in sistema_turnos:
        duracion_turno = 8.0
    else:
        duracion_turno = 24.0

    n_turnos = math.ceil(tiempo_limite / duracion_turno)
    frentes_nombres = list(frentes_info.keys())
    
    avance_por_turno = {frente: {turno: [] for turno in range(1, n_turnos + 1)} for frente in frentes_nombres}
    
    # Se ordena para que la posición N-1 de cada lista corresponda siempre a la
    # simulación N (permite luego seleccionar una simulación específica por índice).
    sim_ids = sorted(df_traza_completa['Simulation_ID'].unique())

    for sim_id in sim_ids:
        df_sim = df_traza_completa[df_traza_completa['Simulation_ID'] == sim_id].copy()
        
        for frente in frentes_nombres:
            # Contar ciclos completados basado en el número de veces que aparece la última secuencia
            df_frente_sim = df_sim[
                (df_sim['Resource_Origin_Tunnel'] == frente) &
                (df_sim['type'] == 'activity')
            ].copy()
            
            if df_frente_sim.empty:
                for turno in range(1, n_turnos + 1):
                    avance_por_turno[frente][turno].append(0)
                continue
            
            df_frente_sim = df_frente_sim.sort_values(by='Finish')
            
            # Agrupar por ciclo y obtener el tiempo de finalización de cada ciclo
            ciclos_finish_times = df_frente_sim.groupby('Cycle')['Finish'].max().sort_values()
            
            for turno in range(1, n_turnos + 1):
                tiempo_fin_turno = turno * duracion_turno
                
                # Contar ciclos completados hasta el fin del turno
                ciclos_completados = (ciclos_finish_times <= tiempo_fin_turno).sum()
                
                avance_ciclo = frentes_info.get(frente, {}).get('metros_por_ciclo', metros_avance)
                if 'Cycle_Advance_Length_m' in df_frente_sim.columns:
                    avances_registrados = df_frente_sim['Cycle_Advance_Length_m'].dropna()
                    avances_registrados = avances_registrados[avances_registrados > 0]
                    if not avances_registrados.empty:
                        avance_ciclo = float(avances_registrados.max())

                avance_acumulado_sim = min(
                    ciclos_completados * avance_ciclo,
                    frentes_info[frente]['distancia']
                )
                
                avance_por_turno[frente][turno].append(avance_acumulado_sim)

    return avance_por_turno


@st.cache_data(show_spinner=False)
def calcular_resumen_bloqueos(_df_traza, run_id):
    """Agrupa los eventos de bloqueo por falta de recurso. Cacheada porque este groupby
    se repetia en cada rerun de la Seccion 17, aunque la traza no cambiara."""
    if _df_traza.empty or 'type' not in _df_traza.columns:
        return None
    df_bloqueos = _df_traza[_df_traza['type'] == 'blocked_no_resource']
    if df_bloqueos.empty:
        return None
    resumen = df_bloqueos.groupby(['Resource_Origin_Tunnel', 'Recurso']).agg(
        Simulaciones_Afectadas=('Simulation_ID', 'nunique'),
        Hora_Bloqueo_Promedio=('Start', 'mean')
    ).reset_index().rename(columns={
        'Resource_Origin_Tunnel': 'Túnel', 'Recurso': 'Recurso Faltante'
    })
    resumen['Hora_Bloqueo_Promedio'] = r2(resumen['Hora_Bloqueo_Promedio'])
    return resumen


@st.cache_data(show_spinner=False)
def calcular_viajes_agregados(_df_tunel, run_id, tunel, sim_id):
    """Agrupa distancia/tiempo de viaje por Simulation_ID y tipo de recurso. Cacheada por
    la misma razon: groupby costoso repetido en cada rerun de la Seccion 18."""
    df_viajes = _df_tunel[_df_tunel['type'] == 'travel'].copy()
    if df_viajes.empty:
        return pd.DataFrame(columns=['Simulation_ID', 'Recurso_Tipo', 'Distancia_Total', 'Tiempo_Total', 'Num_Viajes'])
    # _tipo_recurso_instancia quita solo el sufijo numerico de la instancia ('Jumbo_A_0' ->
    # 'Jumbo_A'). Antes se usaba split('_')[0], que fusionaba recursos distintos
    # (Jumbo_A + Jumbo_F -> 'Jumbo'; Cuadrilla_Minera + Cuadrilla_Carguío -> 'Cuadrilla').
    df_viajes['Recurso_Tipo'] = df_viajes['Recurso'].apply(_tipo_recurso_instancia)
    return df_viajes.groupby(['Simulation_ID', 'Recurso_Tipo']).agg(
        Distancia_Total=('Travel_Distance_m', 'sum'),
        Tiempo_Total=('Travel_Time_Actual', 'sum'),
        Num_Viajes=('Cycle', 'count')
    ).reset_index()


@st.cache_data(show_spinner=False)
def calcular_viajes_por_avance(_df_traza, run_id, tunel, ciclo):
    """Igual que calcular_viajes_agregados, pero solo los viajes hechos por los recursos
    PARA LLEGAR a 'tunel' durante su avance N° 'ciclo'. Cada viaje se registra con el tunel de
    ORIGEN del recurso, pero su 'Ciclo_Frente' es el del tunel de DESTINO (el que se esta
    avanzando); por eso aqui se filtra por destino, igual que la Gantt real por avance."""
    columnas = ['Simulation_ID', 'Recurso_Tipo', 'Distancia_Total', 'Tiempo_Total', 'Num_Viajes']
    if 'Ciclo_Frente' not in _df_traza.columns or 'Resource_Destination_Tunnel' not in _df_traza.columns:
        return pd.DataFrame(columns=columnas)
    df_viajes = _df_traza[
        (_df_traza['type'] == 'travel')
        & (_df_traza['Resource_Destination_Tunnel'] == tunel)
        & (_df_traza['Ciclo_Frente'] == ciclo)
    ].copy()
    if df_viajes.empty:
        return pd.DataFrame(columns=columnas)
    df_viajes['Recurso_Tipo'] = df_viajes['Recurso'].apply(_tipo_recurso_instancia)
    return df_viajes.groupby(['Simulation_ID', 'Recurso_Tipo']).agg(
        Distancia_Total=('Travel_Distance_m', 'sum'),
        Tiempo_Total=('Travel_Time_Actual', 'sum'),
        Num_Viajes=('Cycle', 'count')
    ).reset_index()


# Mantenciones programadas (Menu 12). La mantencion es un evento del EQUIPO (en la traza no tiene
# tunel, 'N/A'). Para verla como una actividad mas de cada tunel se usan DOS representaciones:
#  - 'maintenance' / 'maintenance_failed' (TIPOS_MANTENCION): el evento COMPLETO (ej. 140 h). Se
#    usa solo en la distribucion Monte Carlo (Menú 18), para comparar contra la distribucion de entrada.
#  - 'maintenance_tramo' (TIPO_MANTENCION_TRAMO): el pedazo de esa mantencion que cae DENTRO de
#    cada avance del tunel (ej. 23 h en el avance 1 y el resto en el siguiente). Se usa en el
#    Resumen de Tiempos y en las 3 vistas de la Carta Gantt.
TIPOS_MANTENCION = ['maintenance', 'maintenance_failed']
TIPO_MANTENCION_TRAMO = 'maintenance_tramo'

TIPOS_EVENTO_CICLO_GANTT = [
    'activity', 'delay_fh_oc', 'delay_res', 'delay_bloqueo_total', 'delay_equipment_failure',
    'delay_fleet_change', 'delay_maintenance', 'delay_shift_change', 'blocked_no_resource', TIPO_MANTENCION_TRAMO
]


def _nombre_actividad_mantencion(nombre, recurso):
    """'Jumbo_A_1' + 'Prev 250h' -> "Mantención Programada 'Prev 250h' (Jumbo_A #2)".
    Cada instancia del equipo queda como una actividad distinta (no se suman equipos distintos)."""
    partes = str(recurso).rsplit('_', 1)
    if len(partes) == 2 and partes[1].isdigit():
        equipo = f"{partes[0]} #{int(partes[1]) + 1}"
    else:
        equipo = str(recurso)
    nombre = '' if pd.isna(nombre) else str(nombre)
    return f"Mantención Programada '{nombre}' ({equipo})"


def _mantenciones_de_equipos_usados(df_traza, tunel):
    """Mantenciones programadas (eventos completos) de los equipos que ESE tunel uso en cada
    simulacion: instancias que trabajaron en el tunel, esperaron por mantencion en el, o viajaron
    hacia el. Devuelve el evento completo con su 'Actividad' ya nombrada; vacio si no hay."""
    if df_traza is None or df_traza.empty or 'type' not in df_traza.columns:
        return pd.DataFrame()
    mant = df_traza[df_traza['type'].isin(TIPOS_MANTENCION)]
    if mant.empty:
        return pd.DataFrame()
    propios = (df_traza['Resource_Origin_Tunnel'] == tunel) & df_traza['type'].isin(['activity', 'delay_maintenance'])
    viajes = (df_traza['type'] == 'travel') & (df_traza['Resource_Destination_Tunnel'] == tunel)
    usados = df_traza.loc[propios | viajes, ['Simulation_ID', 'Recurso']].dropna().drop_duplicates()
    usados = usados[usados['Recurso'].astype(str) != 'N/A']
    if usados.empty:
        return pd.DataFrame()
    m = mant.merge(usados, on=['Simulation_ID', 'Recurso'], how='inner')
    if m.empty:
        return pd.DataFrame()
    detalle = m['Detalle_Mantencion'] if 'Detalle_Mantencion' in m.columns else pd.Series('', index=m.index)
    m = m.copy()
    m['Actividad'] = [_nombre_actividad_mantencion(n, r) for n, r in zip(detalle, m['Recurso'])]
    return m


def _restar_intervalos(ini, fin, cortes):
    """[ini, fin] menos la union de 'cortes' (lista de (a, b)). Devuelve los pedazos que quedan."""
    res, cur = [], ini
    for a, b in sorted(cortes):
        if b <= cur or a >= fin:
            continue
        if a > cur:
            res.append((cur, min(a, fin)))
        cur = max(cur, b)
        if cur >= fin:
            break
    if cur < fin:
        res.append((cur, fin))
    return [(x, y) for x, y in res if y - x > 1e-6]


def _tramos_mantencion_por_avance(m, ventanas, tunel, demoras=None, fragmentar=False):
    """Recorta cada mantencion (m) a la ventana de tiempo de cada avance del tunel.
    ventanas: DataFrame con Simulation_ID, Ciclo_Frente, W0, W1 (inicio y fin real del avance).
    Un tramo = parte de la mantencion que cae dentro del avance (ej. mantencion de 140 h que parte
    en el avance 1: 23 h en el avance 1 y el resto en los siguientes).

    SIN DOBLE CONTEO: cuando el tunel esperaba a ese equipo por la mantencion, ese tiempo ya esta
    en la 'Demora por Mantención Programada' (evento 'delay_maintenance', pasado en 'demoras' con
    Simulation_ID, Recurso, Start, Finish). El tramo solo conserva lo que NO esta cubierto por esa
    demora; asi demora + tramo = tiempo total de la mantencion dentro del avance, y ambos se suman
    uno tras otro sin solaparse.
    fragmentar=True (Gantt real): un tramo por cada pedazo restante, con sus horas reales.
    fragmentar=False (resumenes/cascadas): una sola fila por mantencion y avance, con la suma de
    los pedazos como duracion (Finish = Start + suma; Start/Finish no se usan como horas reales)."""
    if m is None or m.empty or ventanas is None or ventanas.empty:
        return pd.DataFrame()
    # (la mantencion no pertenece a ningun avance: se descartan sus columnas de ciclo antes de cruzar)
    t = m.drop(columns=[c for c in ('Ciclo_Frente', 'Secuencia') if c in m.columns]).merge(
        ventanas, on='Simulation_ID', how='inner')
    if t.empty:
        return t
    t = t.copy()
    t['Start'] = np.maximum(t['Start'].astype(float), t['W0'].astype(float))
    t['Finish'] = np.minimum(t['Finish'].astype(float), t['W1'].astype(float))
    t = t[(t['Finish'] - t['Start']) > 1e-6]
    if t.empty:
        return t
    cortes_por = {}
    if demoras is not None and not demoras.empty:
        for (sim, rec), g in demoras.groupby(['Simulation_ID', 'Recurso']):
            cortes_por[(sim, str(rec))] = list(zip(g['Start'].astype(float), g['Finish'].astype(float)))
    filas = []
    for fila in t.to_dict('records'):
        trozos = _restar_intervalos(fila['Start'], fila['Finish'],
                                    cortes_por.get((fila['Simulation_ID'], str(fila['Recurso'])), []))
        if not trozos:
            continue
        if fragmentar:
            for x, y in trozos:
                filas.append({**fila, 'Start': x, 'Finish': y})
        else:
            suma = sum(y - x for x, y in trozos)
            filas.append({**fila, 'Start': trozos[0][0], 'Finish': trozos[0][0] + suma})
    if not filas:
        return pd.DataFrame()
    t = pd.DataFrame(filas)
    t['Start'], t['Finish'] = t['Start'].round(2), t['Finish'].round(2)
    t['Duration'] = (t['Finish'] - t['Start']).round(2)
    t['type'] = TIPO_MANTENCION_TRAMO
    t['Ciclo_Frente'] = t['Ciclo_Frente'].astype(int)
    t['Secuencia'] = 0
    t['Resource_Origin_Tunnel'] = tunel
    t['Es_Personalizada'] = False
    for col in ('Duration_Muestreada', 'Truncado_Horizonte'):
        if col in t.columns:
            t[col] = np.nan
    return t.drop(columns=['W0', 'W1'])


def incorporar_mantenciones_como_actividad(df_tunel, df_traza, tunel):
    """Devuelve df_tunel + las mantenciones programadas de los equipos que ese tunel uso, como una
    actividad mas del tunel:
      - Tramos por avance ('maintenance_tramo'): lo que cae dentro de cada avance y NO esta ya
        contado en la 'Demora por Mantención' (Resumen de Tiempos y Carta Gantt, en sus 3 vistas).
      - Evento completo ('maintenance'/'maintenance_failed', Ciclo_Frente = -1): solo para la
        distribucion Monte Carlo (Menú 18).
    Los reintentos por mantencion fallida se suman bajo la misma actividad del equipo."""
    if df_tunel is None or df_tunel.empty:
        return df_tunel
    m = _mantenciones_de_equipos_usados(df_traza, tunel)
    if m.empty:
        return df_tunel
    ev = df_tunel[df_tunel['type'].isin(TIPOS_EVENTO_CICLO_GANTT) & (df_tunel['type'] != TIPO_MANTENCION_TRAMO)].copy()
    ev['Ciclo_Frente'] = pd.to_numeric(ev.get('Ciclo_Frente'), errors='coerce').fillna(-1).astype(int)
    ev = ev[ev['Ciclo_Frente'] >= 1]
    ventanas = ev.groupby(['Simulation_ID', 'Ciclo_Frente']).agg(
        W0=('Start', 'min'), W1=('Finish', 'max')
    ).reset_index()
    demoras = df_tunel.loc[df_tunel['type'] == 'delay_maintenance', ['Simulation_ID', 'Recurso', 'Start', 'Finish']]
    tramos = _tramos_mantencion_por_avance(m, ventanas, tunel, demoras=demoras, fragmentar=False)
    completas = m.copy()
    completas['Resource_Origin_Tunnel'] = tunel
    completas['Secuencia'] = 0
    completas['Ciclo_Frente'] = -1
    return pd.concat([df_tunel, completas, tramos], ignore_index=True)


def anexar_viajes_destino(df_tunel, df_traza, tunel):
    """df_tunel + los viajes de recursos cuyo DESTINO es este tunel (el viaje se registra con el
    tunel de ORIGEN del recurso). Solo para la opcion 'Incluir viajes' de la Carta Gantt."""
    if df_traza is None or df_traza.empty or 'Resource_Destination_Tunnel' not in df_traza.columns:
        return df_tunel
    extra = df_traza[(df_traza['type'] == 'travel')
                     & (df_traza['Resource_Destination_Tunnel'] == tunel)
                     & (df_traza['Resource_Origin_Tunnel'] != tunel)]
    return df_tunel if extra.empty else pd.concat([df_tunel, extra], ignore_index=True)


# Mapeo de la etiqueta visible al percentil elegido -> nombre de columna que arma
# calcular_datos_gantt (ver PERCENTILES_DISPONIBLES_GANTT mas abajo, usado en la UI).
PERCENTILES_DISPONIBLES_GANTT = {
    'P0 (Mínimo)': 'P0', 'P10': 'P10', 'P30': 'P30', 'P50 (Mediana)': 'P50',
    'Esperanza (Promedio)': 'Esperanza', 'P70': 'P70', 'P90': 'P90', 'P100 (Máximo)': 'P100'
}


# ======================================================================
# CARTA GANTT POR PASO (Secuencia + Actividad) Y POR CICLO DE AVANCE
# ======================================================================
# Antes la Gantt y el "Resumen de Tiempos" agrupaban SOLO por el nombre de la
# actividad. Con un input como el de NH (15 secuencias, pero 'FR' aparece 3 veces:
# secuencias 7, 10 y 15) eso fusionaba las 3 en una sola barra, y mostraba 13 filas en
# vez de 15. Ademas, cuando una secuencia usa 2 recursos en paralelo (TR con
# Cuadrilla_Carguio + Grua, LP, SA, IM, HM, RE) se SUMABAN sus duraciones (horas-recurso)
# en vez de medir cuanto se demora realmente la actividad.
#
# Ahora cada fila es un "PASO" = (Secuencia, Actividad), ordenado por secuencia, y su
# duracion es el tiempo TRANSCURRIDO real (desde que parte el primer recurso hasta que
# termina el ultimo), no la suma de horas-recurso.
#
# Cada evento de la traza trae 'Secuencia' (0 = demora previa al ciclo) y 'Ciclo_Frente'
# (N° de avance del propio tunel: 1, 2, 3...; distinto de 'Cycle', que es un contador
# global compartido por todos los frentes de la simulacion).

def _tipo_recurso_instancia(nombre):
    """'Jumbo_A_0' -> 'Jumbo_A' (quita el sufijo numerico de la instancia)."""
    nombre = str(nombre)
    partes = nombre.rsplit('_', 1)
    if len(partes) == 2 and partes[1].isdigit():
        return partes[0]
    return nombre


def _etiqueta_paso(secuencia, actividad, recurso_tipo=''):
    """'7. FR' para actividades de secuencia; solo el nombre para demoras (secuencia 0).
    Si se indica 'recurso_tipo' (pasos con varios recursos en paralelo) queda
    '12. HM (Grúa)', para mostrar cada recurso por separado."""
    try:
        sec = int(secuencia)
    except (TypeError, ValueError):
        sec = 0
    base = f"{sec}. {actividad}" if sec > 0 else str(actividad)
    return f"{base} ({recurso_tipo})" if recurso_tipo else base


def _color_paso_gantt(actividad, es_personalizada):
    nombre = str(actividad)
    if nombre.startswith('BLOQUEO'):
        return 'black'
    if nombre.startswith('Viaje →'):
        return '#9E9E9E'   # gris: igual que los viajes en la Gantt real
    if nombre.startswith('Mantención Programada') or nombre.startswith('Demora por Mantención'):
        return '#00838F'   # teal: mismo color que la mantención en el Menú 19 / línea de tiempo del recurso
    if nombre.startswith('Demora'):
        return 'red'
    if es_personalizada:
        return 'green'
    return 'rgb(31, 119, 180)'


# ----------------------------------------------------------------------
# Estilo de la Carta Gantt
# ----------------------------------------------------------------------
# Barras de colores suaves por categoría (con leyenda), sin líneas de colores encima, rejilla
# vertical punteada tenue y todas las etiquetas con el MISMO tamaño de letra, a la derecha de
# cada barra (nunca dentro: dentro, Plotly las achica o las gira según el ancho de la barra).
GANTT_MOSTRAR_ETIQUETAS = True   # False = sin números junto a las barras (el valor sigue en el tooltip)
GANTT_TAMANO_LETRA = 12

# color interno del paso (ver _color_paso_gantt) -> categoría que se ve en la leyenda
_CATEGORIA_POR_COLOR_GANTT = {
    'rgb(31, 119, 180)': 'Actividad',
    'green': 'Actividad personalizada',
    '#9E9E9E': 'Viaje',
    'red': 'Demora',
    '#00838F': 'Mantención',
    'black': 'Bloqueo de túnel',
}
# El orden de este diccionario es el orden de la leyenda.
COLORES_CATEGORIA_GANTT = {
    'Actividad': '#5E7A94',
    'Actividad personalizada': '#9BBB6F',
    'Viaje': '#C4C9CF',
    'Demora': '#F27B6B',
    'Mantención': '#4DB6AC',
    'Bloqueo de túnel': '#455A64',
}


def _figura_gantt_base(df, titulo, eje_x_titulo, con_rango=False):
    """Figura Gantt estándar, común a todas las vistas.

    df debe traer: Etiqueta (fila), Base (inicio), Largo (duración), Color (color interno del
    paso) y Hover (texto del tooltip). Con con_rango=True también Err_Plus / Err_Minus, que se
    dibujan como un bigote gris fino (rango P10-P90)."""
    df = df.reset_index(drop=True).copy()
    df['Categoria'] = [_CATEGORIA_POR_COLOR_GANTT.get(str(c), 'Actividad') for c in df['Color']]

    fig = go.Figure()
    for categoria, color in COLORES_CATEGORIA_GANTT.items():
        sub = df[df['Categoria'] == categoria]
        if sub.empty:
            continue
        error_x = None
        if con_rango:
            error_x = dict(
                type='data', symmetric=False, array=sub['Err_Plus'], arrayminus=sub['Err_Minus'],
                color='#9AA5B1', thickness=1, width=3
            )
        fig.add_trace(go.Bar(
            y=sub['Etiqueta'], x=sub['Largo'], base=sub['Base'], orientation='h', name=categoria,
            marker=dict(color=color, line=dict(width=0)),
            customdata=sub['Hover'], hovertemplate='<b>%{y}</b><br>%{customdata}<extra></extra>',
            error_x=error_x
        ))

    fin_barra = df['Base'] + df['Largo']
    fin_visible = fin_barra + df['Err_Plus'] if con_rango else fin_barra
    if GANTT_MOSTRAR_ETIQUETAS:
        fig.add_trace(go.Scatter(
            x=list(fin_visible), y=list(df['Etiqueta']), mode='text',
            text=[f"{fmt2(v)} h" for v in df['Largo']], textposition='middle right',
            textfont=dict(size=GANTT_TAMANO_LETRA), showlegend=False, hoverinfo='skip', cliponaxis=False
        ))
    x_max = float(fin_visible.max()) if len(df) else 0.0
    if x_max <= 0:
        x_max = 1.0

    fig.update_layout(
        barmode='overlay', bargap=0.4,
        title=dict(text=titulo, x=0, xanchor='left', y=0.97, yanchor='top', font=dict(size=14)),
        legend=dict(orientation='h', x=0, xanchor='left', y=1.0, yanchor='bottom'),
        font=dict(size=GANTT_TAMANO_LETRA),
        height=max(280, len(df) * 34 + 170),
        margin=dict(t=100, b=60, l=10, r=20),
        plot_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(align='left'),
        xaxis=dict(
            title=dict(text=eje_x_titulo), range=[0, x_max * 1.14], zeroline=False,
            showgrid=True, gridcolor='rgba(150,150,150,0.30)', griddash='dash',
            showline=True, linecolor='rgba(150,150,150,0.6)'
        ),
        yaxis=dict(
            type='category', autorange='reversed', categoryorder='array',
            categoryarray=list(dict.fromkeys(df['Etiqueta'])), showgrid=False, automargin=True
        ),
    )
    return fig


@st.cache_data(show_spinner=False)
def preparar_eventos_ciclo_tunel(_df_tunel, run_id, tunel, incluir_viajes=False):
    """Eventos de ciclo (actividades y demoras) de un tunel, con 'Secuencia' y
    'Ciclo_Frente' normalizados. Si la traza viene de una corrida anterior a este
    cambio (sin esas columnas), quedan en 0 / -1 y las vistas por ciclo avisan que hay
    que volver a simular."""
    df = _df_tunel[_df_tunel['type'].isin(TIPOS_EVENTO_CICLO_GANTT)].copy()
    if incluir_viajes and 'Resource_Destination_Tunnel' in _df_tunel.columns:
        # Viajes de recursos HACIA este tunel (en la Gantt cada viaje es un paso 'Viaje → ACT (Recurso)'
        # de la misma secuencia que la actividad a la que van).
        viajes = _df_tunel[(_df_tunel['type'] == 'travel')
                           & (_df_tunel['Resource_Destination_Tunnel'] == tunel)].copy()
        if not viajes.empty:
            viajes['Actividad'] = [
                f"Viaje → {a} ({_tipo_recurso_instancia(r)})"
                for a, r in zip(viajes['Actividad'], viajes['Recurso'])
            ]
            df = pd.concat([df, viajes], ignore_index=True)
    if df.empty:
        return df
    for col, defecto in (('Secuencia', 0), ('Ciclo_Frente', -1)):
        if col not in df.columns:
            df[col] = defecto
        df[col] = df[col].fillna(defecto).astype(int)
    if 'Es_Personalizada' in df.columns:
        df['Es_Personalizada'] = df['Es_Personalizada'].fillna(False).astype(bool)
    else:
        df['Es_Personalizada'] = False
    df['Recurso'] = df['Recurso'].fillna('N/A').astype(str)
    return df


def info_avances_tunel(_df_tunel, run_id, tunel, tiempo_limite):
    """Avances disponibles de un tunel para los selectores 'Total / Avance k'.
    Devuelve (ciclos, sims_por_avance, avances_prom):
      ciclos          : lista ordenada de N° de avance con eventos (1, 2, 3...).
      sims_por_avance : Serie {avance: N° de simulaciones que llegaron a el}.
      avances_prom    : avances totales logrados por simulacion en promedio (ej. 4.6 = 4
                        completos + 60% del quinto), o None si no se puede calcular."""
    df_ev = preparar_eventos_ciclo_tunel(_df_tunel, run_id, tunel)
    if df_ev.empty:
        return [], pd.Series(dtype=int), None
    con_avance = df_ev[df_ev['Ciclo_Frente'] >= 1]
    ciclos = sorted(int(c) for c in con_avance['Ciclo_Frente'].unique())
    sims_por_avance = con_avance.groupby('Ciclo_Frente')['Simulation_ID'].nunique()
    estado = calcular_estado_ciclos_tunel(_df_tunel, run_id, tunel, tiempo_limite)
    avances_prom = float(estado.groupby('Simulation_ID')['Fraccion'].sum().mean()) if not estado.empty else None
    return ciclos, sims_por_avance, avances_prom


def formatear_opcion_avance(avance, sims_por_avance, avances_prom):
    """Texto de cada opcion del selector (0 = total del tunel)."""
    if avance == 0:
        extra = f" (~{fmt2(avances_prom)} avances por simulación)" if avances_prom is not None else ""
        return "Total — todos los avances del túnel" + extra
    return f"Avance {avance} ({int(sims_por_avance.get(avance, 0))} simulaciones)"


def _con_tipo_recurso(df_ev):
    """Agrega 'Recurso_Tipo' ('Grúa_0' -> 'Grúa'; '' si el evento no usa recurso)."""
    df = df_ev.copy()
    df['Recurso_Tipo'] = [
        '' if str(r) == 'N/A' else _tipo_recurso_instancia(r) for r in df['Recurso']
    ]
    return df


def _etiquetas_por_paso(claves):
    """claves: lista de (secuencia, actividad, recurso_tipo) -> {clave: etiqueta}.
    El recurso se agrega a la etiqueta SOLO cuando el mismo paso (secuencia + actividad)
    lo ejecutan varios recursos en paralelo; asi cada uno aparece por separado
    (ej. '12. HM (Cuadrilla_Minera)' y '12. HM (Grúa)') y los pasos de un solo recurso
    conservan su etiqueta de siempre ('1. PD')."""
    recursos_por_paso = {}
    for sec, act, rec in claves:
        recursos_por_paso.setdefault((sec, act), set()).add(rec)
    return {
        (sec, act, rec): _etiqueta_paso(sec, act, rec if len(recursos_por_paso[(sec, act)]) > 1 else '')
        for sec, act, rec in claves
    }


def _spans_por_paso_ciclo(df_ev):
    """Una fila por (simulacion, ciclo, secuencia, actividad, TIPO DE RECURSO) con inicio
    real, fin real y tiempo transcurrido de ese recurso. Los recursos que trabajan en
    paralelo en una misma secuencia quedan en filas separadas (no se fusionan)."""
    df = _con_tipo_recurso(df_ev)
    g = df.groupby(['Simulation_ID', 'Ciclo_Frente', 'Secuencia', 'Actividad', 'Recurso_Tipo'], sort=False)
    out = g.agg(
        Start=('Start', 'min'), Finish=('Finish', 'max'),
        Es_Personalizada=('Es_Personalizada', 'max')
    ).reset_index()
    out['Duration'] = out['Finish'] - out['Start']
    return out


@st.cache_data(show_spinner=False)
def calcular_resumen_tiempos_pasos(_df_tunel, run_id, tunel, sim_id, ciclo=None):
    """Tabla 'Resumen de Tiempos': una fila por paso (Secuencia + Actividad) y por recurso,
    ordenada por secuencia, con el tiempo total (suma de todos los ciclos del tunel)
    promediado entre simulaciones (o el de una simulacion puntual si sim_id no es None).
    Con ciclo=k se muestra solo el avance N° k del tunel (lo que duro cada paso en ese avance,
    promediado entre las simulaciones que lo alcanzaron)."""
    columnas = ['Actividad', 'Recursos', 'Duration']
    df_ev = preparar_eventos_ciclo_tunel(_df_tunel, run_id, tunel)
    if df_ev.empty:
        return pd.DataFrame(columns=columnas)
    if sim_id is not None:
        df_ev = df_ev[df_ev['Simulation_ID'] == sim_id]
        if df_ev.empty:
            return pd.DataFrame(columns=columnas)
    spans = _spans_por_paso_ciclo(df_ev)
    if ciclo is not None:
        spans = spans[spans['Ciclo_Frente'] == ciclo]
        if spans.empty:
            return pd.DataFrame(columns=columnas)
    total_sim = spans.groupby(['Simulation_ID', 'Secuencia', 'Actividad', 'Recurso_Tipo'])['Duration'].sum().reset_index()
    prom = total_sim.groupby(['Secuencia', 'Actividad', 'Recurso_Tipo'])['Duration'].mean().reset_index()
    prom = prom.sort_values(['Secuencia', 'Actividad', 'Recurso_Tipo']).reset_index(drop=True)
    claves = list(zip(prom['Secuencia'], prom['Actividad'], prom['Recurso_Tipo']))
    etiquetas = _etiquetas_por_paso(claves)
    prom['Recursos'] = prom['Recurso_Tipo']
    prom['Actividad'] = [etiquetas[k] for k in claves]
    return prom[columnas]


@st.cache_data(show_spinner=False)
def calcular_estado_ciclos_tunel(_df_tunel, run_id, tunel, tiempo_limite):
    """Para cada (simulacion, avance del tunel): cuanto del avance alcanzo a hacerse.
    'Fraccion' = secuencias del ciclo terminadas antes del fin del horizonte / secuencias
    totales del ciclo (ej. 0.5 = el tunel quedo a medio avance cuando se acabo el tiempo).
    Sumar la Fraccion de una simulacion da los 'avances alcanzados' (ej. 3.5)."""
    columnas = ['Simulation_ID', 'Ciclo_Frente', 'Fraccion', 'Completo', 'Inicio', 'Fin']
    df_ev = preparar_eventos_ciclo_tunel(_df_tunel, run_id, tunel)
    if df_ev.empty:
        return pd.DataFrame(columns=columnas)
    df_ev = df_ev[df_ev['Ciclo_Frente'] >= 1]
    if df_ev.empty:
        return pd.DataFrame(columns=columnas)

    acts = df_ev[(df_ev['type'] == 'activity') & (df_ev['Secuencia'] > 0)]
    n_secuencias = max(acts['Secuencia'].nunique(), 1)
    cierre = acts.groupby(['Simulation_ID', 'Ciclo_Frente', 'Secuencia'])['Finish'].max().reset_index()
    cierre['Hecha'] = (cierre['Finish'] < (float(tiempo_limite) - 1e-6)).astype(int)
    hechas = cierre.groupby(['Simulation_ID', 'Ciclo_Frente'])['Hecha'].sum().rename('Hechas').reset_index()

    base = df_ev.groupby(['Simulation_ID', 'Ciclo_Frente']).agg(
        Inicio=('Start', 'min'), Fin=('Finish', 'max')
    ).reset_index()
    base = base.merge(hechas, how='left', on=['Simulation_ID', 'Ciclo_Frente'])
    base['Hechas'] = base['Hechas'].fillna(0)
    base['Fraccion'] = (base['Hechas'] / n_secuencias).clip(upper=1.0)
    base['Completo'] = base['Fraccion'] >= 0.999
    return base[columnas].sort_values(['Simulation_ID', 'Ciclo_Frente']).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def calcular_pasos_gantt(_df_tunel, run_id, tunel, sim_id, modo='acumulado',
                         percentil_barra='P50', ciclo=None, sims_incluidas=None, incluir_viajes=False):
    """Datos de la Gantt en cascada, una fila por PASO (Secuencia + Actividad) y, cuando
    varios recursos lo ejecutan en paralelo, una fila por cada RECURSO (se muestran por
    separado, no fusionados).

    ciclo=None  -> todos los ciclos del tunel:
        modo 'acumulado'      : tiempo total del paso sumando todos los ciclos.
        modo 'promedio_ciclo' : duracion tipica de UNA ocurrencia del paso.
    ciclo=k     -> solo el avance N° k del tunel (una barra por paso con el tiempo que
                   tardo ese paso en ese avance). Con sims_incluidas se restringe a las
                   simulaciones que efectivamente completaron ese avance.
    sim_id      -> una simulacion puntual (deterministica) o None (todas, con percentiles).
    Los pasos de una misma secuencia (recursos/actividades en paralelo) arrancan juntos;
    la secuencia siguiente parte cuando termina la mas larga de ellas.
    """
    df_ev = preparar_eventos_ciclo_tunel(_df_tunel, run_id, tunel, incluir_viajes)
    if df_ev.empty:
        return []
    if sims_incluidas is not None:
        df_ev = df_ev[df_ev['Simulation_ID'].isin(list(sims_incluidas))]
    if sim_id is not None:
        df_ev = df_ev[df_ev['Simulation_ID'] == sim_id]
    if ciclo is not None:
        df_ev = df_ev[df_ev['Ciclo_Frente'] == ciclo]
    if df_ev.empty:
        return []

    spans = _spans_por_paso_ciclo(df_ev)

    if ciclo is None:
        por_sim = spans.groupby(['Simulation_ID', 'Secuencia', 'Actividad', 'Recurso_Tipo']).agg(
            Duration=('Duration', 'sum'),
            N_Ciclos_Paso=('Ciclo_Frente', 'nunique'),
            Es_Personalizada=('Es_Personalizada', 'max')
        ).reset_index()
        if modo == 'promedio_ciclo':
            ciclos_por_sim = spans[(spans['Secuencia'] > 0) & (spans['Ciclo_Frente'] > 0)].groupby(
                'Simulation_ID')['Ciclo_Frente'].nunique()
            es_actividad = (por_sim['Secuencia'] > 0) & ~por_sim['Actividad'].astype(str).str.startswith('BLOQUEO')
            divisor = pd.Series(
                np.where(es_actividad, por_sim['N_Ciclos_Paso'], por_sim['Simulation_ID'].map(ciclos_por_sim).fillna(0)),
                index=por_sim.index
            )
            por_sim = por_sim[divisor > 0].copy()
            por_sim['Duration'] = por_sim['Duration'] / divisor[divisor > 0]
            if por_sim.empty:
                return []
    else:
        por_sim = spans

    filas = []
    grupos_paso = list(por_sim.groupby(['Secuencia', 'Actividad', 'Recurso_Tipo'], sort=True))
    etiquetas = _etiquetas_por_paso([k for k, _ in grupos_paso])
    for (sec, act, rec), grp in grupos_paso:
        vals = grp['Duration'].values.astype(float)
        pct = percentiles_reporte(vals)
        filas.append({
            'Actividad': etiquetas[(sec, act, rec)], 'Secuencia': int(sec),
            'Recursos': rec,
            'P0': pct['p0'], 'P10': float(np.percentile(vals, 10)), 'P30': pct['p30'],
            'P50': float(np.percentile(vals, 50)), 'Esperanza': pct['esperanza'],
            'P70': pct['p70'], 'P90': float(np.percentile(vals, 90)), 'P100': pct['p100'],
            'N_Sims': int(len(vals)),
            'Color': _color_paso_gantt(act, bool(grp['Es_Personalizada'].max())),
            '_act': str(act), '_rec': str(rec),
        })

    # Cascada: las demoras (secuencia 0) van una tras otra; los pasos de una misma
    # secuencia parten juntos y la secuencia siguiente arranca al terminar la mas larga.
    # Con viajes: la actividad de un recurso parte cuando termina SU viaje (el viaje y la actividad
    # del mismo recurso van en serie; los de recursos distintos siguen en paralelo).
    viaje_previo = {
        (f['Secuencia'], f['_act'], f['_rec']): float(f.get(percentil_barra, f['P50']))
        for f in filas if f['_act'].startswith('Viaje →')
    }
    # Igual con la espera por mantencion: la actividad de ese equipo parte cuando termina su demora
    # (demora tras demora, no una barra encima de la otra).
    espera_mant_previa = {}
    for f in filas:
        if f['_act'].startswith('Demora por Mantención') and f['Secuencia'] > 0:
            k = (f['Secuencia'], f['_rec'])
            espera_mant_previa[k] = espera_mant_previa.get(k, 0.0) + float(f.get(percentil_barra, f['P50']))
    cursor, seq_abierta, inicio_seq, fin_seq = 0.0, None, 0.0, 0.0
    for fila in filas:
        valor = float(fila.get(percentil_barra, fila['P50']))
        if fila['Secuencia'] == 0:
            fila['Start_Barra'] = cursor
            cursor += valor
        else:
            if fila['Secuencia'] != seq_abierta:
                if seq_abierta is not None:
                    cursor = fin_seq
                seq_abierta, inicio_seq, fin_seq = fila['Secuencia'], cursor, cursor
            espera_viaje = 0.0
            es_previa = fila['_act'].startswith(('Viaje →', 'Demora por Mantención'))
            if viaje_previo and not es_previa:
                espera_viaje = viaje_previo.get(
                    (fila['Secuencia'], f"Viaje → {fila['_act']} ({fila['_rec']})", fila['_rec']), 0.0)
            if espera_mant_previa and not es_previa:
                espera_viaje += espera_mant_previa.get((fila['Secuencia'], fila['_rec']), 0.0)
            fila['Start_Barra'] = inicio_seq + espera_viaje
            fin_seq = max(fin_seq, inicio_seq + espera_viaje + valor)
        fila['Valor_Barra'] = valor
    return filas


@st.cache_data(show_spinner=False)
def obtener_eventos_ciclo_real(_df_traza, run_id, tunel, sim_id, ciclo, incluir_viajes):
    """Eventos REALES (con sus horas de inicio/fin) de un avance puntual de un tunel en
    una simulacion puntual: actividades y demoras, y opcionalmente los viajes de
    recursos hacia ese tunel (el viaje se registra con el tunel de ORIGEN del recurso,
    por eso se filtra por tunel de DESTINO)."""
    sub = _df_traza[_df_traza['Simulation_ID'] == sim_id]
    if 'Ciclo_Frente' not in sub.columns or sub.empty:
        return pd.DataFrame()
    ev = sub[(sub['Resource_Origin_Tunnel'] == tunel)
             & sub['type'].isin(TIPOS_EVENTO_CICLO_GANTT)
             & (sub['Ciclo_Frente'] == ciclo)]
    partes = [ev]
    if incluir_viajes:
        partes.append(sub[(sub['type'] == 'travel')
                          & (sub['Resource_Destination_Tunnel'] == tunel)
                          & (sub['Ciclo_Frente'] == ciclo)])
    df = pd.concat(partes, ignore_index=True)
    if df.empty:
        return df
    # Mantenciones programadas de los equipos que uso este tunel: el tramo de cada una que cae
    # dentro de la ventana de ESTE avance (ej. 23 h de una mantencion de 140 h), como una actividad mas.
    if not ev.empty:
        m = _mantenciones_de_equipos_usados(sub, tunel)
        if not m.empty:
            ventana = pd.DataFrame([{
                'Simulation_ID': sim_id, 'Ciclo_Frente': ciclo,
                'W0': float(ev['Start'].min()), 'W1': float(ev['Finish'].max())
            }])
            demoras = ev.loc[ev['type'] == 'delay_maintenance', ['Simulation_ID', 'Recurso', 'Start', 'Finish']]
            tramos = _tramos_mantencion_por_avance(m, ventana, tunel, demoras=demoras, fragmentar=True)
            if not tramos.empty:
                tramos['Recurso'] = 'N/A'   # el nombre del equipo ya va en la actividad
                df = pd.concat([df, tramos], ignore_index=True)
    df['Secuencia'] = df['Secuencia'].fillna(0).astype(int)
    df['Recurso'] = df['Recurso'].fillna('N/A').astype(str)
    df['Es_Personalizada'] = df['Es_Personalizada'].fillna(False).astype(bool) if 'Es_Personalizada' in df.columns else False
    return df.sort_values(['Secuencia', 'Start']).reset_index(drop=True)


def figura_gantt_cascada(gantt_data, titulo, eje_x_titulo, determinista, percentil_col, etiqueta_modo):
    """Gantt en cascada por paso. Con varias simulaciones agrega el rango P10-P90 (bigote gris fino)."""
    df = pd.DataFrame(gantt_data)
    df['Etiqueta'] = df['Actividad']
    df['Base'] = df['Start_Barra']
    df['Largo'] = df['Valor_Barra']

    if determinista:
        df['Hover'] = df.apply(
            lambda r: f"{r['Recursos'] or 'sin recurso'}<br>Duración: {fmt2(r['Valor_Barra'])} h", axis=1)
    else:
        df['Hover'] = df.apply(lambda r: (
            f"{r['Recursos'] or 'sin recurso'} — {int(r['N_Sims'])} simulaciones<br>"
            f"Largo de la barra ({percentil_col}): {fmt2(r['Valor_Barra'])} h<br>"
            f"P0: {fmt2(r['P0'])}h | P10: {fmt2(r['P10'])}h | P30: {fmt2(r['P30'])}h<br>"
            f"P50: {fmt2(r['P50'])}h | Esperanza: {fmt2(r['Esperanza'])}h | P70: {fmt2(r['P70'])}h<br>"
            f"P90: {fmt2(r['P90'])}h | P100: {fmt2(r['P100'])}h<br>{etiqueta_modo}"
        ), axis=1)
        df['Err_Plus'] = (df['P90'] - df['Valor_Barra']).clip(lower=0)
        df['Err_Minus'] = (df['Valor_Barra'] - df['P10']).clip(lower=0)
    return _figura_gantt_base(df, titulo, eje_x_titulo, con_rango=not determinista)


def figura_gantt_ciclo_real(df_ev, inicio_ciclo, titulo, duracion_turno, limite_rel=None,
                            ref_pasos=None, ref_col=None):
    """Gantt REAL de un avance: cada barra va de su hora de inicio a su hora de fin
    (relativas al inicio del avance), una fila por actividad/recurso, en orden de secuencia.

    ref_pasos / ref_col: si vienen, el tooltip de cada actividad agrega el percentil de referencia
    y en qué percentil quedó esta simulación (ya no se dibuja ninguna línea en la carta).
    limite_rel: se conserva por compatibilidad con quien llama; el fin del horizonte simulado se
    informa en un texto bajo la carta, no como línea."""
    etiquetas, vistas = [], {}
    for _, r in df_ev.iterrows():
        if r['type'] == 'travel':
            base = f"{_etiqueta_paso(r['Secuencia'], 'Viaje')} → {r['Actividad']} ({_tipo_recurso_instancia(r['Recurso'])})"
        elif r['Recurso'] != 'N/A':
            base = f"{_etiqueta_paso(r['Secuencia'], r['Actividad'])} ({_tipo_recurso_instancia(r['Recurso'])})"
        else:
            base = _etiqueta_paso(r['Secuencia'], r['Actividad'])
        vistas[base] = vistas.get(base, 0) + 1
        etiquetas.append(base if vistas[base] == 1 else f"{base} #{vistas[base]}")

    df = df_ev.copy().reset_index(drop=True)
    df['Etiqueta'] = etiquetas
    df['Base'] = df['Start'] - inicio_ciclo
    df['Largo'] = (df['Finish'] - df['Start']).clip(lower=0)
    df['Color'] = [
        '#9E9E9E' if t == 'travel' else _color_paso_gantt(a, bool(p))
        for t, a, p in zip(df['type'], df['Actividad'], df['Es_Personalizada'])
    ]
    hover = df.apply(lambda r: (
        f"Inicio: {fmt2(r['Start'])} h (turno {int(r['Start'] // duracion_turno) + 1})<br>"
        f"Fin: {fmt2(r['Finish'])} h<br>Duración: {fmt2(r['Largo'])} h"
    ), axis=1)

    extra = [''] * len(df)
    if ref_pasos and ref_col:
        # Dónde cae esta simulación respecto de la misma actividad en las demás simulaciones que
        # completaron este avance. Solo texto del tooltip.
        for i, r in df.iterrows():
            if r['type'] != 'activity':
                continue
            clave = (int(r['Secuencia']), str(r['Actividad']),
                     '' if str(r['Recurso']) == 'N/A' else _tipo_recurso_instancia(r['Recurso']))
            vals = ref_pasos.get(clave)
            if vals is None or len(vals) == 0:
                continue
            p_val = float(percentiles_reporte(vals)[ref_col.lower()])
            rank = float((np.asarray(vals) <= float(r['Largo']) + 1e-9).mean() * 100)
            extra[i] = (f"<br>{ref_col} de esta actividad: {fmt2(p_val)} h"
                        f"<br>Esta simulación: ≈P{rank:.0f} ({len(vals)} sims)")
    df['Hover'] = [h + e for h, e in zip(hover, extra)]
    return _figura_gantt_base(df, titulo, 'Horas desde el inicio de este avance')


# ----------------------------------------------------
# Muestras Monte Carlo por PASO y RECURSO (para la Menú 18)
# ----------------------------------------------------
# Tipos de evento cuya duracion se muestra en la distribucion Monte Carlo.
TIPOS_EVENTO_MC = [
    'activity', 'delay', 'delay_fh_oc', 'delay_res', 'delay_bloqueo_total',
    'delay_equipment_failure', 'delay_fleet_change', 'delay_maintenance', 'delay_shift_change'
] + TIPOS_MANTENCION


def preparar_muestras_mc(df_tunel):
    """Prepara las duraciones de un tunel (todas las simulaciones) para la distribucion
    Monte Carlo, separadas por PASO (Secuencia + Actividad) y por TIPO DE RECURSO.
    Asi, si una secuencia tiene recursos en paralelo (ej. 5. SA con Cuadrilla_Minera y
    con Grua), cada uno queda por separado y sus muestras NO se mezclan ni se acumulan;
    y una actividad repetida (ej. FR en las secuencias 7, 10 y 15) tampoco se fusiona.

    Devuelve (df_mc, claves, etiquetas):
      df_mc     : eventos con 'Secuencia', 'Ciclo_Frente' y 'Recurso_Tipo' normalizados.
      claves    : lista ordenada de (secuencia, actividad, recurso_tipo); las demoras
                  (secuencia 0) van primero, luego los pasos en orden de secuencia.
      etiquetas : {clave: texto visible}, ej. '12. HM (Grúa)'.
    """
    vacio = (pd.DataFrame(), [], {})
    if df_tunel is None or df_tunel.empty or 'type' not in df_tunel.columns \
            or 'Actividad' not in df_tunel.columns or 'Duration' not in df_tunel.columns:
        return vacio
    df = df_tunel[df_tunel['type'].isin(TIPOS_EVENTO_MC)].copy()
    df = df[df['Actividad'].notna() & df['Duration'].notna()]
    if df.empty:
        return vacio
    df['Actividad'] = df['Actividad'].astype(str)
    for col, defecto in (('Secuencia', 0), ('Ciclo_Frente', -1)):
        if col not in df.columns:
            df[col] = defecto
        df[col] = df[col].fillna(defecto).astype(int)
    if 'Recurso' in df.columns:
        df['Recurso'] = df['Recurso'].fillna('N/A').astype(str)
    else:
        df['Recurso'] = 'N/A'
    df = _con_tipo_recurso(df)
    # Cambio de turno: una sola muestra por (simulacion, tunel, ventana), no una por cada espera parcial.
    if 'Ventana_Inicio' in df.columns:
        _es_ct = df['type'] == 'delay_shift_change'
        df = pd.concat([
            df[~_es_ct],
            df[_es_ct].drop_duplicates(subset=['Simulation_ID', 'Resource_Origin_Tunnel', 'Ventana_Inicio'])
        ], ignore_index=True)
    claves = sorted(set(zip(df['Secuencia'], df['Actividad'], df['Recurso_Tipo'])))
    return df, claves, _etiquetas_por_paso(claves)


def muestras_paso_mc(df_mc, clave, avance=None):
    """Duraciones (una por ocurrencia, en todas las simulaciones) de UN paso y UN recurso.
    avance=None -> todos los avances del tunel (total); avance=k -> solo el avance N° k."""
    sec, act, rec = clave
    mascara = (df_mc['Secuencia'] == sec) & (df_mc['Actividad'] == act) & (df_mc['Recurso_Tipo'] == rec)
    if avance is not None:
        mascara &= (df_mc['Ciclo_Frente'] == avance)
    sub = df_mc.loc[mascara]
    # Para actividades se usa la duracion MUESTREADA completa ('Duration_Muestreada'): asi el
    # resultado se compara 1 a 1 con la distribucion de entrada (y su rango truncado). 'Duration'
    # queda cortada cuando el limite de tiempo de la simulacion interrumpe la actividad.
    if 'Duration_Muestreada' in sub.columns:
        vals = sub['Duration_Muestreada'].where(sub['Duration_Muestreada'].notna(), sub['Duration'])
    else:
        vals = sub['Duration']
    return vals.astype(float).values


def contar_truncados_horizonte(df_mc, clave, avance=None):
    """Cuantas de las ocurrencias de ese paso fueron cortadas por el limite de tiempo de la
    simulacion (su 'Duration' en la traza es menor a la duracion muestreada)."""
    if 'Truncado_Horizonte' not in df_mc.columns:
        return 0
    sec, act, rec = clave
    mascara = (df_mc['Secuencia'] == sec) & (df_mc['Actividad'] == act) & (df_mc['Recurso_Tipo'] == rec)
    if avance is not None:
        mascara &= (df_mc['Ciclo_Frente'] == avance)
    return int((df_mc.loc[mascara, 'Truncado_Horizonte'] == True).sum())


# ----------------------------------------------------------------------
# Estilo del dashboard "Distribución de Resultados Monte Carlo" (Menú 18)
# ----------------------------------------------------------------------
_CSS_TARJETAS = (
    "<style>"
    ".mc-grid{display:flex;flex-wrap:wrap;justify-content:center;gap:12px;margin:6px 0 18px 0;}"
    ".mc-card{flex:1 1 110px;box-sizing:border-box;text-align:center;padding:14px 8px;"
    "border-radius:12px;border:1px solid rgba(128,128,128,0.25);background:rgba(128,128,128,0.07);}"
    ".mc-card.mc-dest{box-shadow:inset 0 4px 0 #5E7A94;background:rgba(94,122,148,0.12);}"
    ".mc-label{font-size:0.78rem;opacity:0.72;margin-bottom:4px;white-space:nowrap;}"
    ".mc-valor{font-size:1.65rem;font-weight:600;line-height:1.15;white-space:nowrap;}"
    ".mc-unidad{font-size:0.85rem;font-weight:400;opacity:0.65;margin-left:3px;}"
    ".mc-nota{font-size:0.78rem;opacity:0.7;margin-top:6px;}"
    "@media (max-width:1280px){.mc-grid.mc-muchas .mc-card{flex:0 0 calc(33.333% - 8px);"
    "max-width:none !important;}}"
    "</style>"
)


def mostrar_tarjetas_metricas(items, ancho_max=170):
    """Tarjetas de métricas del mismo tamaño, con el texto centrado y el grupo completo centrado en
    la página (reemplaza a st.metric, que queda pegado a la izquierda y se desalinea entre filas).

    items = [(etiqueta, valor, unidad, destacada[, nota]), ...]. La nota es una línea chica bajo el
    valor (ej. 'Máximo: 6485'). ancho_max (px) es el ancho máximo de cada tarjeta: súbelo cuando hay
    pocas tarjetas. Con muchas (más de 4), en pantallas angostas pasan a una cuadrícula de 3 por fila."""
    tarjetas = []
    for item in items:
        etiqueta, valor, unidad, destacada = item[:4]
        nota = item[4] if len(item) > 4 else ''
        clase = 'mc-card mc-dest' if destacada else 'mc-card'
        u = f'<span class="mc-unidad">{unidad}</span>' if unidad else ''
        n = f'<div class="mc-nota">{nota}</div>' if nota else ''
        tarjetas.append(
            f'<div class="{clase}" style="max-width:{int(ancho_max)}px">'
            f'<div class="mc-label">{etiqueta}</div><div class="mc-valor">{valor}{u}</div>{n}</div>'
        )
    clase_grid = 'mc-grid mc-muchas' if len(items) > 4 else 'mc-grid'
    st.markdown(_CSS_TARJETAS + f'<div class="{clase_grid}">' + ''.join(tarjetas) + '</div>',
                unsafe_allow_html=True)


def mostrar_titulo_centrado(titulo, subtitulo=''):
    """Título (y subtítulo opcional) centrados, para encabezar un grupo de tarjetas."""
    sub = f'<div style="opacity:0.7;font-size:0.85rem;margin-top:2px;">{subtitulo}</div>' if subtitulo else ''
    st.markdown(
        f'<div style="text-align:center;margin-bottom:8px;"><div style="font-weight:600;">{titulo}</div>{sub}</div>',
        unsafe_allow_html=True)


# ----------------------------------------------------
# Dashboard de resultados (Menu 17): Avance final, tiempo por tunel y reparto de recursos
# ----------------------------------------------------
def _botones_descarga_tabla(df, nombre_archivo, clave):
    """Botones para descargar una tabla en CSV y en Excel."""
    col_csv, col_xlsx, _ = st.columns([1, 1, 4])
    with col_csv:
        st.download_button(
            "Descargar CSV", df.to_csv(index=False).encode('utf-8-sig'),
            file_name=f"{nombre_archivo}.csv", mime="text/csv", key=f"dl_csv_{clave}"
        )
    with col_xlsx:
        try:
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as escritor:
                df.to_excel(escritor, sheet_name=str(nombre_archivo)[:31], index=False)
            st.download_button(
                "Descargar Excel", buffer.getvalue(),
                file_name=f"{nombre_archivo}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"dl_xlsx_{clave}"
            )
        except Exception:
            pass  # sin openpyxl, queda disponible el CSV


def _selector_vista(clave):
    """Selector Gráfico / Tabla de un bloque del dashboard. Devuelve True si se pide la tabla."""
    vista = st.radio(
        "Formato", ["Gráfico", "Tabla"], horizontal=True,
        key=f"vista_{clave}", label_visibility="collapsed"
    )
    return vista == "Tabla"


def dashboard_avance_final(sim):
    """Avance final por tunel: tarjetas KPI, % de la meta por tunel (barras; con barras de error
    P10-P90 en modo promedio) y tabla descargable. Devuelve el DataFrame de la tabla (lo usa el
    export a Excel). sim=None -> promedio de todas las simulaciones; sim=N -> simulacion N."""
    filas, pct_esp, err_sup, err_inf, nombres = [], [], [], [], []
    avance_total, meta_total = 0.0, 0.0
    n_sims = 0
    for frente, avances in st.session_state.resultados.items():
        if not avances:
            continue
        dist = st.session_state.frentes_info[frente]['distancia']
        n_sims = max(n_sims, len(avances))
        if sim is None:
            arr = np.asarray(avances, dtype=float)
            p = percentiles_reporte(arr)
            esperanza = float(np.mean(arr))
            fila = {
                'Túnel': frente, 'Distancia Total (m)': r2(dist),
                'P0 - Mínimo (m)': p['p0'], 'P10 (m)': p['p10'], 'P30 (m)': p['p30'],
                'P50 (m)': p['p50'], 'Esperanza (m)': p['esperanza'], 'P70 (m)': p['p70'],
                'P90 (m)': p['p90'], 'P100 - Máximo (m)': p['p100'],
                'Completado (Esperanza, %)': r2(esperanza / dist * 100) if dist else 0.0,
            }
            valor = esperanza
            if dist:
                err_sup.append(max(0.0, float(np.percentile(arr, 90)) - esperanza) / dist * 100)
                err_inf.append(max(0.0, esperanza - float(np.percentile(arr, 10))) / dist * 100)
            else:
                err_sup.append(0.0); err_inf.append(0.0)
        else:
            idx = min(sim - 1, len(avances) - 1)
            valor = float(avances[idx])
            fila = {
                'Túnel': frente, 'Distancia Total (m)': r2(dist),
                f'Avance Sim. {sim} (m)': r2(valor),
                'Porcentaje Completado (%)': r2(valor / dist * 100) if dist else 0.0,
            }
        filas.append(fila)
        nombres.append(frente)
        pct_esp.append(valor / dist * 100 if dist else 0.0)
        avance_total += valor
        meta_total += dist

    df = pd.DataFrame(filas)
    if df.empty:
        st.info("No hay resultados de avance para mostrar.")
        return df

    # Tarjetas KPI
    pct_meta = avance_total / meta_total * 100 if meta_total else 0.0
    i_rez = int(np.argmin(pct_esp))
    etiqueta_valor = "Avance total (promedio)" if sim is None else f"Avance total (Sim. {sim})"
    mostrar_tarjetas_metricas([
        (etiqueta_valor, fmt2(avance_total), "m", True, f"Meta: {fmt2(meta_total)} m"),
        ("% de la meta", fmt2(pct_meta), "%", False),
        ("Túnel más rezagado", str(nombres[i_rez]), "", False, f"{fmt2(pct_esp[i_rez])}% completado"),
        ("Simulaciones", f"{n_sims}" if sim is None else f"{sim} de {n_sims}", "", False),
    ], ancho_max=250)

    ver_tabla = _selector_vista("avance_final")
    if ver_tabla:
        mostrar_tabla(df, width='stretch', hide_index=True)
        _botones_descarga_tabla(df, "avance_final_por_tunel" if sim is None else f"avance_final_sim_{sim}", "avance_final")
    else:
        colores = ['#2e7d32' if v >= 100 else '#1f77b4' for v in pct_esp]
        fig = go.Figure(go.Bar(
            x=nombres, y=pct_esp, marker_color=colores,
            text=[f"{fmt2(v)}%" for v in pct_esp], textposition='outside',
            error_y=(dict(type='data', symmetric=False, array=err_sup, arrayminus=err_inf,
                          color='rgba(120,120,120,0.9)') if sim is None else None),
            hovertemplate="%{x}<br>%{y:.2f}% de la meta<extra></extra>",
        ))
        fig.add_hline(y=100, line_dash='dash', line_color='rgba(120,120,120,0.8)',
                      annotation_text="Meta", annotation_position="top left")
        titulo = ("% de la meta completado por túnel (Esperanza, barras P10 a P90)" if sim is None
                  else f"% de la meta completado por túnel (Simulación N° {sim})")
        fig.update_layout(
            title=dict(text=titulo, x=0.5, xanchor='center', font=dict(size=14)),
            yaxis_title="% de la meta", xaxis_title="Túnel", height=420,
            margin=dict(t=70, b=50, l=60, r=20), plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(rangemode='tozero', gridcolor='rgba(150,150,150,0.25)'),
        )
        st.plotly_chart(fig, width='stretch', key="grafico_pct_meta_tunel")
    return df


# Categorias del grafico "En que se va el tiempo": (etiqueta, tipos de evento de la traza, color).
# Productivo y Viaje son horas-recurso (los recursos en paralelo se suman); el resto son esperas del
# tunel (horas que el frente quedo detenido por ese motivo).
_CATEGORIAS_TIEMPO = [
    ('Productivo', ['activity'], '#1f77b4'),
    ('Viaje', ['travel'], '#ff7f0e'),
    ('Demora por restricción', ['delay', 'delay_fh_oc', 'delay_res'], '#7f7f7f'),
    ('Falla de equipo', ['delay_equipment_failure'], '#d62728'),
    ('Mantención programada', ['delay_maintenance'], '#9467bd'),
    ('Cambio de turno', ['delay_shift_change'], '#e377c2'),
    ('Cambio de recurso', ['delay_fleet_change'], '#17becf'),
    ('Bloqueo / sin recurso', ['delay_bloqueo_total', 'blocked_no_resource'], '#8c564b'),
]


def _horas_por_tunel_y_recurso(df_traza, sim, frentes, percentil='Esperanza'):
    """Horas de la traza por tunel y categoria (productivo, viaje y todas las demoras), promediadas por
    simulacion (o de una sola). Devuelve (df_tiempo [Túnel + una columna '<categoria> (h)' por cada
    categoria con horas > 0], df_recurso_tunel [filas=tipo de recurso, columnas=túneles]).
    percentil: solo aplica con sim=None. 'Esperanza' = promedio entre simulaciones; 'P0','P10',...,'P100' =
    ese percentil ENTRE simulaciones, calculado por separado para cada categoria y tunel. El mapa de recursos
    (df_recurso_tunel) siempre es el promedio."""
    vacio = (pd.DataFrame(), pd.DataFrame())
    if df_traza is None or df_traza.empty or 'type' not in df_traza.columns:
        return vacio
    tipos = [t for _, ts, _ in _CATEGORIAS_TIEMPO for t in ts]
    d = df_traza[df_traza['type'].isin(tipos)]
    ids_sims = sorted(df_traza['Simulation_ID'].unique())
    if sim is None:
        n = max(1, len(ids_sims))
    else:
        d = d[d['Simulation_ID'] == sim]
        n = 1
    if d.empty:
        return vacio
    d = d.copy()
    # El viaje se cuenta en el tunel de destino; todo lo demas, en el tunel donde ocurre.
    d['Tunel'] = np.where(d['type'] == 'travel', d['Resource_Destination_Tunnel'], d['Resource_Origin_Tunnel'])
    d['Tipo_Recurso'] = d['Recurso'].map(_tipo_recurso_instancia)
    tuneles = [f for f in frentes if f in set(d['Tunel'])]
    if not tuneles:
        return vacio
    df_t = pd.DataFrame({'Túnel': tuneles})
    for etiqueta, ts, _ in _CATEGORIAS_TIEMPO:
        if sim is None and percentil != 'Esperanza':
            # Horas de cada simulacion (0 si esa simulacion no tuvo la categoria) -> percentil entre simulaciones.
            por_sim = (d[d['type'].isin(ts)].groupby(['Simulation_ID', 'Tunel'])['Duration'].sum()
                       .unstack('Tunel').reindex(ids_sims).fillna(0.0))
            q = float(percentil[1:])   # 'P90' -> 90
            valores = [float(np.percentile(por_sim[t].values, q)) if t in por_sim.columns else 0.0
                       for t in tuneles]
        else:
            horas = d[d['type'].isin(ts)].groupby('Tunel')['Duration'].sum() / n
            valores = [float(horas.get(t, 0.0)) for t in tuneles]
        if etiqueta in ('Productivo', 'Viaje') or sum(valores) > 1e-9:
            df_t[f'{etiqueta} (h)'] = valores
    df_t['Total (h)'] = df_t.drop(columns=['Túnel']).sum(axis=1)
    mat = (d[d['type'] == 'activity'].groupby(['Tipo_Recurso', 'Tunel'])['Duration'].sum() / n).unstack(fill_value=0.0)
    mat = mat.reindex(columns=[f for f in frentes if f in mat.columns]).fillna(0.0)
    return df_t, mat


def dashboard_tiempo_recursos(sim, df_resumen_recursos=None):
    """Composicion del tiempo por tunel (Viaje vs Productivo) y reparto de recursos entre tunieles
    (mapa de calor). Cada bloque tiene vista Gráfico / Tabla con descarga."""
    frentes = list(st.session_state.frentes_info.keys())
    percentil = 'Esperanza'
    if sim is None:
        etiqueta_pct = st.selectbox(
            "Valor a mostrar entre todas las simulaciones", list(PERCENTILES_DISPONIBLES_GANTT.keys()),
            index=list(PERCENTILES_DISPONIBLES_GANTT.keys()).index('Esperanza (Promedio)'),
            key="percentil_utilizacion_recursos",
            help="Esperanza es el promedio. P10 = en el 10% de las simulaciones las horas fueron iguales o menores "
                 "(escenario favorable en horas); P90 = en el 90% fueron iguales o menores (escenario exigente). "
                 "Cada categoría y túnel se calcula por separado, por lo que la suma de las barras de un percentil "
                 "no es el percentil del total. Aplica al gráfico y tabla de '¿En qué se va el tiempo de cada túnel?'.")
        percentil = PERCENTILES_DISPONIBLES_GANTT[etiqueta_pct]
    df_t, mat = _horas_por_tunel_y_recurso(st.session_state.get('df_traza_eventos_cache'), sim, frentes, percentil)
    if sim is not None:
        sufijo = f"Simulación N° {sim}"
        sufijo_mapa = sufijo
    else:
        sufijo = "promedio por simulación" if percentil == 'Esperanza' else f"{percentil} entre simulaciones"
        sufijo_mapa = "promedio por simulación"
    if df_t.empty:
        st.info("No hay eventos en la traza para mostrar tiempos y recursos.")
        if df_resumen_recursos is not None and not df_resumen_recursos.empty:
            mostrar_tabla(df_resumen_recursos, width='stretch', hide_index=True)
        return

    # --- En que se va el tiempo de cada tunel ---
    st.markdown("**¿En qué se va el tiempo de cada túnel?**")
    ayuda_inline("Productivo y Viaje son horas-evento: los recursos en paralelo se suman y el viaje cuenta en el "
                 "túnel de destino. Las demás categorías (restricciones, fallas, mantenciones programadas, cambios "
                 "de recurso, bloqueos) son horas que el túnel estuvo detenido por ese motivo, así que no son "
                 f"directamente comparables con las horas-recurso. Valores: {sufijo}.", "Cómo leer estas horas")
    if _selector_vista("tiempo_tunel"):
        mostrar_tabla(df_t.round(2), width='stretch', hide_index=True)
        _botones_descarga_tabla(df_t.round(2), "tiempo_por_tunel", "tiempo_tunel")
    else:
        orden = df_t.sort_values('Total (h)')
        fig_t = go.Figure()
        for etiqueta, _, color in _CATEGORIAS_TIEMPO:
            col = f'{etiqueta} (h)'
            if col in orden.columns:
                fig_t.add_bar(y=orden['Túnel'], x=orden[col], name=etiqueta, orientation='h', marker_color=color)
        fig_t.update_layout(
            barmode='stack', height=max(260, 60 + 38 * len(orden)),
            xaxis_title=f"Horas-evento ({sufijo})", yaxis_title="Túnel",
            margin=dict(t=40, b=50, l=60, r=20), plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation='h', x=0.5, xanchor='center', y=1.02, yanchor='bottom'),
            xaxis=dict(gridcolor='rgba(150,150,150,0.25)'),
        )
        st.plotly_chart(fig_t, width='stretch', key="grafico_tiempo_tunel")

    # --- Como se reparten los recursos entre tuneles ---
    st.markdown("**¿Cómo se reparten los recursos entre túneles?**")
    ayuda_inline(f"Horas productivas de cada tipo de recurso en cada túnel ({sufijo_mapa}).", "Cómo leer el mapa")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        como_pct = st.toggle("Ver como % de las horas de cada recurso", key="toggle_recursos_pct")
    with col_b:
        ver_tabla_rec = _selector_vista("recursos_tunel")

    if mat.empty:
        st.info("No hay horas productivas de recursos para mostrar.")
    else:
        datos = mat.div(mat.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0) * 100 if como_pct else mat
        unidad = "%" if como_pct else "h"
        if ver_tabla_rec:
            df_m = datos.round(2).reset_index().rename(columns={'Tipo_Recurso': 'Recurso'})
            df_m.columns.name = None
            mostrar_tabla(df_m, width='stretch', hide_index=True)
            _botones_descarga_tabla(df_m, "recursos_por_tunel_pct" if como_pct else "recursos_por_tunel_horas", "recursos_tunel")
        else:
            fig_h = go.Figure(go.Heatmap(
                z=datos.values, x=list(datos.columns), y=list(datos.index),
                colorscale='Blues', colorbar=dict(title=unidad),
                text=np.round(datos.values, 1), texttemplate="%{text}",
                hovertemplate="Recurso: %{y}<br>Túnel: %{x}<br>%{z:.2f} " + unidad + "<extra></extra>",
            ))
            fig_h.update_layout(
                height=max(260, 120 + 40 * len(datos.index)), xaxis_title="Túnel", yaxis_title="Recurso",
                margin=dict(t=30, b=50, l=80, r=20), plot_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig_h, width='stretch', key="grafico_recursos_tunel")

    # Resumen de utilizacion por recurso (antes era la unica tabla de esta seccion)
    if df_resumen_recursos is not None and not df_resumen_recursos.empty:
        with st.expander("Utilización por recurso (tabla)", expanded=False):
            mostrar_tabla(df_resumen_recursos, width='stretch', hide_index=True)
            _botones_descarga_tabla(df_resumen_recursos, "utilizacion_de_recursos", "util_recursos")


def estilo_figura_mc(fig, titulo, x_titulo, y_titulo, con_leyenda, rango_x=None):
    """Mismo diseño para los dos gráficos del dashboard: título y leyenda centrados, misma altura y
    márgenes (así los dos recuadros quedan a la par, con o sin leyenda) y rejilla tenue."""
    suave = 'rgba(150,150,150,0.25)'
    fig.update_layout(
        title=dict(text=titulo, x=0.5, xanchor='center', y=0.96, yanchor='top', font=dict(size=14)),
        height=420, margin=dict(t=90, b=55, l=60, r=20), font=dict(size=12),
        plot_bgcolor='rgba(0,0,0,0)', showlegend=con_leyenda,
        legend=dict(orientation='h', x=0.5, xanchor='center', y=1.0, yanchor='bottom'),
        xaxis=dict(title=dict(text=x_titulo), rangemode='tozero', showgrid=True, gridcolor=suave,
                   griddash='dash', zeroline=False, showline=True, linecolor='rgba(150,150,150,0.6)'),
        yaxis=dict(title=dict(text=y_titulo), rangemode='tozero', showgrid=True, gridcolor=suave,
                   zeroline=False),
    )
    if rango_x:
        fig.update_xaxes(range=rango_x)
    return fig


def buscar_distribucion_falla(nombre_actividad, tunel, escenarios_falla):
    """Distribucion de entrada (ya con su rango truncado) de una 'Demora por Falla de Equipo
    (Recurso)' segun los escenarios del Menu 11. Devuelve {'Tiempo','Distribucion'} o None.
    Si varios escenarios aplican al tunel, usa el primero que tenga distribucion para ese recurso."""
    m = re.match(r'^Demora por Falla de Equipo \((.+)\)$', str(nombre_actividad))
    if not m:
        return None
    recurso = m.group(1)
    for esc in reversed(list(escenarios_falla or [])):   # el motor deja el ULTIMO escenario por tunel
        if tunel not in esc.get('frentes_aplicables', []):
            continue
        cfg = esc.get('fallas_por_recurso', {}).get(recurso)
        if cfg and cfg.get('modo_demora') == 'Distribución estadística' \
                and isinstance(cfg.get('params_distribucion'), dict):
            return {'Tiempo': cfg['params_distribucion'], 'Distribucion': cfg['distribucion']}
    return None


def buscar_distribucion_mantencion(nombre_actividad, mantenciones):
    """Distribucion de entrada (con su rango truncado '_trunc_lo'/'_trunc_hi') de una
    "Mantención Programada 'X' (Recurso #N)" segun la configuracion del Menu 12. Devuelve
    {'Tiempo','Distribucion'} o None (duracion manual/constante o mantencion no encontrada)."""
    m = re.match(r"^Mantención Programada '(.*)' \((.+) #(\d+)\)$", str(nombre_actividad))
    if not m:
        return None
    nombre, recurso, n_inst = m.group(1), m.group(2), int(m.group(3))
    for cfg in mantenciones or []:
        if str(cfg.get('nombre')) != nombre or cfg.get('recurso') != recurso \
                or int(cfg.get('instancia_id', -1)) + 1 != n_inst:
            continue
        if cfg.get('modo_duracion') == 'Distribución estadística' \
                and cfg.get('distribucion') not in (None, 'Cte') \
                and isinstance(cfg.get('params_distribucion'), dict):
            return {'Tiempo': cfg['params_distribucion'], 'Distribucion': cfg['distribucion']}
        return None
    return None


def buscar_distribucion_cambio_turno(nombre_actividad, config_turno):
    """Distribucion de entrada (con su rango truncado) de 'Demora por Cambio de Turno' (Menu 5)."""
    if str(nombre_actividad) != "Demora por Cambio de Turno" or not config_turno or not config_turno.get('activo'):
        return None
    if config_turno.get('distribucion') == 'Cte' or not isinstance(config_turno.get('params'), dict):
        return None
    return {'Tiempo': config_turno['params'], 'Distribucion': config_turno['distribucion']}


def buscar_distribucion_entrada(df_actividades_input, clave):
    """Fila de la hoja Actividades (con 'Tiempo' y 'Distribucion') que corresponde EXACTAMENTE
    a ese paso y recurso, para superponerla al resultado simulado. None si no existe
    (ej. demoras por restricciones o fallas, que no tienen distribucion de entrada)."""
    sec, act, rec = clave
    if sec <= 0 or df_actividades_input is None or df_actividades_input.empty \
            or 'Actividad' not in df_actividades_input.columns:
        return None
    df = df_actividades_input[df_actividades_input['Actividad'].astype(str) == act]
    if 'Secuencia' in df.columns:
        df = df[pd.to_numeric(df['Secuencia'], errors='coerce') == sec]
    if rec and 'Recurso' in df.columns:
        con_recurso = df[df['Recurso'].astype(str) == rec]
        if not con_recurso.empty:
            df = con_recurso
    return None if df.empty else df.iloc[0]


# ----------------------------------------------------
# Traza de un tunel ordenada por AVANCE (seccion 15)
# ----------------------------------------------------
def preparar_traza_tunel_por_avance(df_traza, sim_id, tunel):
    """Eventos de una simulacion donde 'tunel' es origen o destino, con una columna 'Avance'
    (N° de avance del PROPIO tunel: 1, 2, 3...) y ordenados por avance y, dentro de cada
    avance, cronologicamente (demoras -> viajes de llegada -> actividades -> viajes de salida).

    Como se asigna el avance:
      * Actividades, demoras y viajes de LLEGADA al tunel: su 'Ciclo_Frente' (exacto).
      * Viajes de SALIDA (el recurso se va a otro tunel): su 'Ciclo_Frente' es el del tunel
        de destino, asi que se asignan al avance de este tunel que estaba en curso (o recien
        terminado) cuando el recurso partio.
      * Eventos sin 'Ciclo_Frente' (ej. mantenciones o cambios de flota): igual que los de salida.
    'Avance' = 0 solo si el tunel no tiene ningun avance con eventos en esa simulacion.
    Agrega ademas 'Sec.' (N° de secuencia del paso, o '—' en demoras y viajes de salida)."""
    df = df_traza[
        (df_traza['Simulation_ID'] == sim_id)
        & ((df_traza['Resource_Origin_Tunnel'] == tunel) | (df_traza['Resource_Destination_Tunnel'] == tunel))
    ].copy()
    if df.empty:
        df['Avance'] = pd.Series(dtype=int)
        df['Sec.'] = pd.Series(dtype=str)
        return df

    ciclo = pd.to_numeric(df['Ciclo_Frente'], errors='coerce') if 'Ciclo_Frente' in df.columns \
        else pd.Series(np.nan, index=df.index)
    es_salida = (
        (df['type'] == 'travel')
        & (df['Resource_Origin_Tunnel'] == tunel)
        & (df['Resource_Destination_Tunnel'] != tunel)
    )
    exacto = ~es_salida & ciclo.notna() & (ciclo >= 1)
    df['Avance'] = np.where(exacto, ciclo, 0)
    df['Avance'] = df['Avance'].fillna(0).astype(int)

    # Hora en que parte cada avance del tunel (primer evento propio de ese avance).
    inicios = df.loc[exacto].groupby('Avance')['Start'].min().sort_index()
    pendientes = df['Avance'] == 0
    if pendientes.any() and not inicios.empty:
        horas_inicio = inicios.values
        numeros = inicios.index.values
        # cantidad de avances que ya habian partido ESTRICTAMENTE antes del evento
        n_previos = np.searchsorted(horas_inicio, df.loc[pendientes, 'Start'].values, side='left')
        df.loc[pendientes, 'Avance'] = numeros[np.maximum(n_previos - 1, 0)]

    if 'Secuencia' in df.columns:
        sec = pd.to_numeric(df['Secuencia'], errors='coerce')
    else:
        sec = pd.Series(np.nan, index=df.index)
    df['Sec.'] = np.where(es_salida | sec.isna() | (sec <= 0), '—', sec.fillna(0).astype(int).astype(str))

    def _orden_tipo(t):
        t = str(t)
        if t.startswith('delay') or t.startswith('maintenance'):
            return 0
        return 1 if t == 'travel' else 2
    df['_orden_tipo'] = df['type'].map(_orden_tipo)
    df = df.sort_values(['Avance', 'Start', '_orden_tipo'], kind='mergesort').drop(columns='_orden_tipo')
    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def graficar_geometria(df_frentes, df_fh, df_oc, df_res, radio_restriccion, radios_por_tipo=None):
    """Grafica los túneles y las restricciones, incluyendo círculos de impacto.

    Cacheada con @st.cache_data: es una función pura (arma un go.Figure solo a partir
    de sus argumentos, sin leer session_state ni mutar sus entradas), y antes vivía
    fuera de cualquier @st.fragment, por lo que se reconstruía en cada rerun de la app
    aunque nada de la geometría hubiera cambiado. La firma de cache incluye todos los
    parámetros de los que depende el resultado (frentes, restricciones, radio), así que
    se invalida sola apenas cambie cualquiera de ellos."""
    
    data_g = []
    
    # 1. TÚNELES (Frentes)
    for _, row in df_frentes.iterrows():
        data_g.append(
            go.Scatter(
                x=[row['Xi'], row['Xf']], y=[row['Yi'], row['Yf']],
                mode='lines+markers',
                name=f"{row['Frentes']} (Túnel)",
                line=dict(width=3, color='green'),
                marker=dict(size=6, symbol='circle', color='green')
            )
        )
        
    # 2. Restricciones y sus radios
    restricciones = [
        (df_fh, 'FH (Frente Hundimiento)', 'red', 'circle'),
        (df_oc, 'OC (Obras Civiles)', 'blue', 'square'),
        (df_res, 'R (Restricción por Turno)', 'darkorange', 'star')
    ]
    
    legend_added = set()

    for df, name, color, symbol in restricciones:
        if 'X' in df.columns and 'Y' in df.columns and not df.empty:
            
            # Graficar los centros de restricción
            data_g.append(
                go.Scatter(
                    x=df['X'], y=df['Y'],
                    mode='markers',
                    name=name,
                    marker=dict(symbol=symbol, size=10, color=color),
                    text=df.apply(lambda r: f"{name}<br>X: {r['X']}<br>Y: {r['Y']}", axis=1),
                    hoverinfo='text'
                )
            )

            # Graficar el círculo de impacto para cada restricción
            if radio_restriccion > 0:
                for i, row in df.iterrows():
                    Px, Py = row['X'], row['Y']
                    radio_evento = radio_restriccion
                    if 'Tipo' in row.index:
                        radio_evento = restriction_radius(row.get('Tipo'), radio_restriccion, radios_por_tipo)
                    t = np.linspace(0, 2 * np.pi, 100)
                    circle_x = Px + radio_evento * np.cos(t)
                    circle_y = Py + radio_evento * np.sin(t)
                    
                    show_legend_for_radius = False
                    if name not in legend_added:
                        show_legend_for_radius = True
                        legend_added.add(name)

                    data_g.append(
                        go.Scatter(
                            x=circle_x, y=circle_y,
                            mode='lines',
                            line=dict(width=1, color=color, dash='dash'),
                            name=f'Radio de Impacto ({name.split()[0]})',
                            showlegend=show_legend_for_radius
                        )
                    )


    fig = go.Figure(data=data_g)
    fig.update_layout(
        title='Geometría del Nivel: Túneles y Restricciones',
        xaxis_title='Coordenada X',
        yaxis_title='Coordenada Y',
        legend_title="Elementos",
        hovermode="closest",
        width=None, height=600,
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )
    return fig


# ============================================================
# VISTA 3D DE TÚNELES: sección real extruida, avances por color y esferas de restricción
# ============================================================
# Cada túnel se dibuja como un prisma a ESCALA REAL: la sección (ancho x alto, forma de
# portal con techo en arco) se extruye a lo largo del eje Xi,Yi -> Xf,Yf.
#   - modo 'Geometría' (Menú 15): el túnel completo, de un solo color, sin cortes de avance.
#   - modo 'Por avance' (Menú 17): lo construido dividido en avances (largo = metros por
#     ciclo), un color por avance (Avance 1, 2, 3...).
#   - modo 'Total' (Menú 17): lo construido de un solo color por túnel.
# Las restricciones (FH, OC, R) se dibujan como ESFERAS translúcidas con su radio de impacto.
# Son funciones puras (solo usan numpy/plotly), sin session_state.

PALETA_AVANCES_3D = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#17becf', '#bcbd22', '#f2b705',
]
COLOR_TUNEL_GEOMETRIA_3D = '#2e7d32'
MODO_GEOMETRIA_3D = "Geometría"
MODO_POR_AVANCE_3D = "Por avance (Avance 1, 2, 3...)"
MODO_TOTAL_3D = "Total (un color por túnel)"
# MODO_GEOLOGIA_3D se define junto al motor MWD


def parsear_dimensiones_seccion(texto):
    """Lee 'ancho x alto' desde un texto tipo '4.2x4.1', '4,0 X 4,5' o 'Sección 4×4'.
    Devuelve (ancho, alto) en metros o None si el texto no tiene ese formato."""
    if texto is None:
        return None
    m = re.search(r'(\d+(?:[.,]\d+)?)\s*[xX×*]\s*(\d+(?:[.,]\d+)?)', str(texto))
    if not m:
        return None
    try:
        ancho = float(m.group(1).replace(',', '.'))
        alto = float(m.group(2).replace(',', '.'))
    except ValueError:
        return None
    if ancho <= 0 or alto <= 0:
        return None
    return ancho, alto


def perfil_seccion_tunel(ancho, alto, n_arco=17):
    """Contorno cerrado de la sección en coordenadas locales (u = lateral, v = altura).
    Origen en el centro del piso. Piso plano, hastiales verticales y techo en arco circular
    (flecha = 25% del ancho, máx. 50% del alto). Devuelve array (M, 2)."""
    c = ancho / 2.0
    flecha = min(0.25 * ancho, 0.5 * alto)
    radio = (c * c + flecha * flecha) / (2.0 * flecha)
    th0 = math.acos(min(1.0, c / radio))
    th = np.linspace(th0, math.pi - th0, n_arco)
    arco = np.column_stack([radio * np.cos(th), alto - radio + radio * np.sin(th)])
    base = np.array([[-c, 0.0], [c, 0.0]], dtype=float)
    return np.vstack([base, arco])  # el lazo se cierra solo (último punto -> primero)


def _marco_tunel(xi, yi, xf, yf):
    dx, dy = xf - xi, yf - yi
    largo = math.hypot(dx, dy)
    if largo < 1e-9:
        return None
    return largo, (dx / largo, dy / largo), (-dy / largo, dx / largo)


def _anillo_3d(origen, d, n, s, perfil):
    """Sección ubicada en la estación s (m desde la boca) -> array (M, 3)."""
    u, v = perfil[:, 0], perfil[:, 1]
    x = origen[0] + d[0] * s + n[0] * u
    y = origen[1] + d[1] * s + n[1] * u
    return np.column_stack([x, y, v])


def _malla_tramo(origen, d, n, s0, s1, perfil, tapa_final=True):
    """Superficie lateral (piso + hastiales + techo) del tramo s0..s1 y, opcionalmente,
    una tapa en s1 (la cara del frente). Devuelve (vértices, caras)."""
    m = len(perfil)
    a0 = _anillo_3d(origen, d, n, s0, perfil)
    a1 = _anillo_3d(origen, d, n, s1, perfil)
    idx = np.arange(m)
    j = (idx + 1) % m
    caras = [np.column_stack([idx, j, m + idx]), np.column_stack([j, m + j, m + idx])]
    vertices = [a0, a1]
    if tapa_final:
        centro = a1.mean(axis=0, keepdims=True)  # interior del contorno (convexo)
        vertices.append(centro)
        caras.append(np.column_stack([np.full(m, 2 * m), m + idx, m + j]))
    return np.vstack(vertices), np.vstack(caras)


def _malla_esfera(cx, cy, cz, radio, n_lat=12, n_lon=24):
    """Esfera (malla lat-lon) centrada en (cx, cy, cz). Devuelve (vértices, caras)."""
    lat = np.linspace(0.0, math.pi, n_lat + 1)
    lon = np.linspace(0.0, 2.0 * math.pi, n_lon, endpoint=False)
    th, ph = np.meshgrid(lat, lon, indexing='ij')
    verts = np.column_stack([
        (cx + radio * np.sin(th) * np.cos(ph)).ravel(),
        (cy + radio * np.sin(th) * np.sin(ph)).ravel(),
        (cz + radio * np.cos(th)).ravel(),
    ])
    i, j = np.meshgrid(np.arange(n_lat), np.arange(n_lon), indexing='ij')
    a = (i * n_lon + j).ravel()
    b = (i * n_lon + (j + 1) % n_lon).ravel()
    c = ((i + 1) * n_lon + j).ravel()
    d = ((i + 1) * n_lon + (j + 1) % n_lon).ravel()
    caras = np.vstack([np.column_stack([a, b, c]), np.column_stack([b, d, c])])
    return verts, caras


def _nuevo_acum():
    return {'verts': [], 'faces': [], 'text': [], 'n': 0}


def _agregar_malla(acum, verts, faces, texto):
    acum['verts'].append(verts)
    acum['faces'].append(faces + acum['n'])
    acum['text'].extend([texto] * len(verts))
    acum['n'] += len(verts)


def _traza_mesh(acum, color, nombre, opacidad=1.0, grupo=None, suave=False):
    v = np.vstack(acum['verts'])
    f = np.vstack(acum['faces'])
    return go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
        color=color, opacity=opacidad, name=nombre, legendgroup=grupo or nombre,
        showlegend=True, hovertext=acum['text'], hoverinfo='text',
        flatshading=not suave, lighting=dict(ambient=0.55, diffuse=0.9, specular=0.1),
    )


def graficar_tuneles_3d(tuneles, titulo="Túneles en 3D", modo_color=MODO_GEOMETRIA_3D,
                        mostrar_pendiente=True, mostrar_lineas=True, restricciones=None):
    """Figura Plotly 3D de los túneles, a escala real.

    tuneles: lista de dicts con: nombre, xi, yi, xf, yf, ancho, alto (m),
             metros_ciclo (largo de cada avance, m) y avance (m construidos desde la boca).
    modo_color: 'Geometría' (túnel completo, un color, sin avances; ignora 'avance'),
                'Por avance' (un color por N° de avance) o 'Total' (un color por túnel).
    restricciones: lista opcional de dicts {df, name, color, radios}; df con columnas X, Y.
                   'radios' (lista o None) = radio de impacto de cada punto; si viene, se
                   dibuja una esfera translúcida de ese radio centrada en el punto (a nivel
                   del piso, Z = 0).
    """
    modo = str(modo_color)
    es_geo = modo.startswith("Geom")
    es_total = modo.startswith("Total")
    es_geol = modo.startswith("Geolog")
    por_clase = {}
    por_avance, por_tunel, avance_real = {}, {}, {}
    pendiente = _nuevo_acum()
    lx, ly, lz = [], [], []
    etq = {'x': [], 'y': [], 'z': [], 't': []}

    for t_idx, t in enumerate(tuneles):
        marco = _marco_tunel(t['xi'], t['yi'], t['xf'], t['yf'])
        if marco is None:
            continue
        largo, d, n = marco
        origen = (t['xi'], t['yi'])
        perfil = perfil_seccion_tunel(t['ancho'], t['alto'])
        info_sec = f"Sección {t['ancho']:g} × {t['alto']:g} m"

        if es_geo:
            avance_real[t_idx] = largo
            acum = por_tunel.setdefault(t_idx, _nuevo_acum())
            v, f = _malla_tramo(origen, d, n, 0.0, largo, perfil, tapa_final=True)
            _agregar_malla(acum, v, f, f"{t['nombre']}<br>Largo: {largo:.1f} m<br>{info_sec}")
        else:
            avance = max(0.0, min(float(t.get('avance') or 0.0), largo))
            avance_real[t_idx] = avance
            m = float(t.get('metros_ciclo') or 0.0)
            if m > 1e-9 and avance > 0:
                nseg = int(math.ceil(avance / m - 1e-9))
                limites = [min(k * m, avance) for k in range(nseg + 1)]
            else:
                limites = [0.0, avance] if avance > 1e-9 else [0.0]

            if es_geol:
                for d0, h0, c in (t.get('geologia') or [(0.0, largo, 'Sin MWD')]):
                    a0, a1 = max(float(d0), 0.0), min(float(h0), avance)
                    if a1 - a0 < 1e-9:
                        continue
                    acum = por_clase.setdefault(c, _nuevo_acum())
                    v, f = _malla_tramo(origen, d, n, a0, a1, perfil, tapa_final=(a1 >= avance - 1e-9))
                    _agregar_malla(acum, v, f, f"{t['nombre']}<br>Roca {c}: {float(d0):.1f} → {float(h0):.1f} m"
                                               f"<br>{info_sec}")
                estaciones = limites if avance > 1e-9 else []
            elif es_total:
                if avance > 1e-9:
                    acum = por_tunel.setdefault(t_idx, _nuevo_acum())
                    v, f = _malla_tramo(origen, d, n, 0.0, avance, perfil, tapa_final=True)
                    _agregar_malla(acum, v, f,
                                   f"{t['nombre']}<br>Construido: {avance:.1f} m de {largo:.1f} m"
                                   f"<br>{info_sec}")
                estaciones = [0.0, avance] if avance > 1e-9 else []
            else:
                ultimo = len(limites) - 1
                for k in range(1, len(limites)):
                    s0, s1 = limites[k - 1], limites[k]
                    if s1 - s0 < 1e-9:
                        continue
                    acum = por_avance.setdefault(k, _nuevo_acum())
                    v, f = _malla_tramo(origen, d, n, s0, s1, perfil, tapa_final=(k == ultimo))
                    _agregar_malla(acum, v, f,
                                   f"{t['nombre']}<br>Avance {k}: {s0:.1f} → {s1:.1f} m "
                                   f"({s1 - s0:.1f} m)<br>{info_sec}")
                estaciones = limites if avance > 1e-9 else []

            if mostrar_pendiente and avance < largo - 1e-6:
                v, f = _malla_tramo(origen, d, n, avance, largo, perfil, tapa_final=True)
                _agregar_malla(pendiente, v, f,
                               f"{t['nombre']}<br>Pendiente: {avance:.1f} → {largo:.1f} m<br>{info_sec}")
                estaciones = list(estaciones) + ([largo] if estaciones else [0.0, largo])

            if mostrar_lineas:
                for s in sorted(set(round(e, 6) for e in estaciones)):
                    anillo = _anillo_3d(origen, d, n, s, perfil)
                    anillo = np.vstack([anillo, anillo[:1]])
                    lx += list(anillo[:, 0]) + [None]
                    ly += list(anillo[:, 1]) + [None]
                    lz += list(anillo[:, 2]) + [None]

        etq['x'].append(origen[0]); etq['y'].append(origen[1])
        etq['z'].append(t['alto'] * 1.6); etq['t'].append(str(t['nombre']))

    trazas = []
    if pendiente['n'] > 0:
        trazas.append(_traza_mesh(pendiente, '#b0b0b0', 'Pendiente (por construir)', 0.18))
    if es_geo:
        for t_idx, acum in por_tunel.items():
            trazas.append(_traza_mesh(acum, COLOR_TUNEL_GEOMETRIA_3D,
                                      f"{tuneles[t_idx]['nombre']} (Túnel)"))
    elif es_geol:
        for c in list(MWD_CLASES) + ['Sin MWD']:
            if c in por_clase:
                trazas.append(_traza_mesh(por_clase[c], MWD_COLORES.get(c, '#78909c'), f"Roca {c}"))
    elif es_total:
        for t_idx, acum in por_tunel.items():
            trazas.append(_traza_mesh(
                acum, PALETA_AVANCES_3D[t_idx % len(PALETA_AVANCES_3D)],
                f"{tuneles[t_idx]['nombre']} (total {avance_real[t_idx]:.1f} m)"))
    else:
        for k in sorted(por_avance):
            trazas.append(_traza_mesh(
                por_avance[k], PALETA_AVANCES_3D[(k - 1) % len(PALETA_AVANCES_3D)], f"Avance {k}"))

    if lx:
        trazas.append(go.Scatter3d(x=lx, y=ly, z=lz, mode='lines', name='Límites de avance',
                                   line=dict(color='#222222', width=2), hoverinfo='skip'))
    if etq['t']:
        trazas.append(go.Scatter3d(x=etq['x'], y=etq['y'], z=etq['z'], mode='text',
                                   text=etq['t'], textposition='top center',
                                   name='Nombre del túnel', showlegend=False, hoverinfo='skip'))

    # Restricciones: centro (marcador) + esfera translúcida con el radio de impacto
    for r in (restricciones or []):
        df_r = r['df']
        if df_r is None or df_r.empty or not {'X', 'Y'}.issubset(df_r.columns):
            continue
        xs = df_r['X'].astype(float).to_numpy()
        ys = df_r['Y'].astype(float).to_numpy()
        trazas.append(go.Scatter3d(
            x=xs, y=ys, z=np.zeros(len(xs)), mode='markers', name=r['name'], legendgroup=r['name'],
            marker=dict(size=4, color=r['color'], symbol=r.get('symbol', 'circle')),
            hovertemplate=r['name'] + "<br>X: %{x}<br>Y: %{y}<extra></extra>"))
        radios = r.get('radios')
        if radios is not None:
            acum_esf = _nuevo_acum()
            for x0, y0, rad in zip(xs, ys, radios):
                if rad is None or not np.isfinite(rad) or rad <= 0:
                    continue
                v, f = _malla_esfera(x0, y0, 0.0, float(rad))
                _agregar_malla(acum_esf, v, f, f"{r['name']}<br>Radio de impacto: {rad:g} m")
            if acum_esf['n'] > 0:
                trazas.append(_traza_mesh(acum_esf, r['color'],
                                          f"Esfera de impacto ({r['name'].split()[0]})",
                                          0.12, grupo=r['name'], suave=True))

    fig = go.Figure(data=trazas)
    fig.update_layout(
        title=titulo, height=700, margin=dict(l=0, r=0, t=50, b=0),
        legend_title_text='Elementos' if es_geo else ('Calidad de roca' if es_geol else ('Túneles' if es_total else 'Avances')),
        scene=dict(
            aspectmode='data',
            xaxis_title='Coordenada X (m)', yaxis_title='Coordenada Y (m)', zaxis_title='Altura (m)',
            camera=dict(eye=dict(x=1.3, y=-1.6, z=0.9)),
        ),
    )
    return fig



# ---------------- Interfaz Streamlit de la vista 3D ----------------
DIM_MIN_SECCION_3D = 4.0  # las secciones van de 4 x 4 m hacia arriba


def _base_tuneles_desde_df(df_frentes):
    """Datos mínimos de cada frente a partir de la hoja Frentes (antes de simular)."""
    seccion_col = columna_seccion(df_frentes)
    metros_sec = st.session_state.get('metros_por_seccion', {}) or {}
    metros_def = float(st.session_state.get('metros_avance', 4.0))
    base = []
    for _, row in df_frentes.iterrows():
        sec = str(row[seccion_col]) if seccion_col else ''
        base.append({
            'nombre': row['Frentes'],
            'xi': float(row['Xi']), 'yi': float(row['Yi']),
            'xf': float(row['Xf']), 'yf': float(row['Yf']),
            'distancia': calcular_distancia(row['Xi'], row['Yi'], row['Xf'], row['Yf']),
            'seccion': sec,
            'metros_base': float(metros_sec.get(sec, metros_def)),
        })
    return base


def _base_tuneles_desde_info(frentes_info):
    """Datos mínimos de cada frente a partir de frentes_info (lo que usó la simulación)."""
    metros_def = float(st.session_state.get('metros_avance', 4.0))
    return [{
        'nombre': f, 'xi': float(i['xi']), 'yi': float(i['yi']),
        'xf': float(i['xf']), 'yf': float(i['yf']),
        'distancia': float(i['distancia']), 'seccion': i.get('seccion', ''),
        'metros_base': float(i.get('metros_por_ciclo', metros_def)),
    } for f, i in frentes_info.items()]


def _usa_dimensiones_por_seccion():
    """True si en el Menú 5 está activado 'Usar Metros por Ciclo variable por Sección':
    en ese caso cada túnel trae su dimensión desde su Sección (ej. '4.2x4.1')."""
    return bool(st.session_state.get('usar_metros_variables_seccion', False))


def _tuneles_sin_dimension_en_seccion(base):
    """Túneles cuyas dimensiones NO vienen de la Sección y por lo tanto las define el usuario
    en el Menú 15: todos si el interruptor del Menú 5 está apagado; si está activado, solo
    los que tienen una Sección sin formato 'AnchoxAlto'."""
    if not _usa_dimensiones_por_seccion():
        return [b['nombre'] for b in base]
    return [b['nombre'] for b in base if parsear_dimensiones_seccion(b['seccion']) is None]


def _dimensiones_3d(base):
    """{túnel: (ancho, alto, origen)}. Con el interruptor del Menú 5 activado, sale de la
    Sección de cada túnel; si no, de lo que el usuario definió en el Menú 15 (4 x 4 m por
    defecto, de ahí hacia arriba)."""
    ancho_m = max(DIM_MIN_SECCION_3D, float(st.session_state.get('dim3d_ancho', DIM_MIN_SECCION_3D)))
    alto_m = max(DIM_MIN_SECCION_3D, float(st.session_state.get('dim3d_alto', DIM_MIN_SECCION_3D)))
    usa_seccion = _usa_dimensiones_por_seccion()
    dims = {}
    for b in base:
        d = parsear_dimensiones_seccion(b['seccion']) if usa_seccion else None
        dims[b['nombre']] = (d[0], d[1], 'Sección (Parámetros Generales)') if d else (ancho_m, alto_m, 'Definida en Geometría')
    return dims


def _entradas_dimension_manual():
    """Ancho y alto definidos por el usuario (mínimo 4 x 4 m). Se guardan en claves propias
    de session_state para que persistan aunque el usuario vuelva a la vista 2D, y para que
    los resultados (Menú 17) usen exactamente las mismas dimensiones."""
    c1, c2 = st.columns(2)
    ancho = c1.number_input(
        "Ancho de la sección (m)", min_value=DIM_MIN_SECCION_3D, step=0.1, key="geo3d_ancho_w",
        value=max(DIM_MIN_SECCION_3D, float(st.session_state.get('dim3d_ancho', DIM_MIN_SECCION_3D))))
    alto = c2.number_input(
        "Alto de la sección (m)", min_value=DIM_MIN_SECCION_3D, step=0.1, key="geo3d_alto_w",
        value=max(DIM_MIN_SECCION_3D, float(st.session_state.get('dim3d_alto', DIM_MIN_SECCION_3D))))
    st.session_state.dim3d_ancho = float(ancho)
    st.session_state.dim3d_alto = float(alto)


def _armar_tuneles_3d(base, avances, dims):
    """Combina geometría + avance + dimensiones en la lista que usa graficar_tuneles_3d.
    'avances' = {frente: metros} o None (vista de geometría: no se usa el avance). El largo
    de cada avance es el MISMO que usa el motor: metros por ciclo del frente (por sección)
    afectado por el castigo de la restricción geológica aplicable."""
    mapa_geo = {}
    for res_geo in st.session_state.get('restricciones_geologicas', []):
        for f in res_geo.get('frentes_aplicables', []):
            mapa_geo.setdefault(f, res_geo)  # igual que el motor: gana la primera
    tuneles = []
    for b in base:
        metros = avance_efectivo_frente(
            b['nombre'], b['metros_base'], {b['nombre']: {'metros_por_ciclo': b['metros_base']}}, mapa_geo
        )
        ancho, alto, origen = dims[b['nombre']]
        tuneles.append({
            'nombre': b['nombre'], 'xi': b['xi'], 'yi': b['yi'], 'xf': b['xf'], 'yf': b['yf'],
            'ancho': ancho, 'alto': alto, 'metros_ciclo': metros, 'origen_dim': origen,
            'avance': b['distancia'] if avances is None else float(avances.get(b['nombre'], 0.0)),
        })
    return tuneles


def _radios_restriccion(df, radio_restriccion, radios_por_tipo):
    """Radio de impacto de cada punto, con el MISMO criterio que el gráfico 2D: el radio
    general, o el radio por tipo (Polvorazo, PA FH...) si la tabla trae columna 'Tipo'."""
    if df is None or df.empty or not {'X', 'Y'}.issubset(df.columns) or radio_restriccion <= 0:
        return None
    if 'Tipo' in df.columns:
        return [restriction_radius(t, radio_restriccion, radios_por_tipo) for t in df['Tipo']]
    return [float(radio_restriccion)] * len(df)


def _tabla_dimensiones(tuneles):
    return pd.DataFrame([{
        'Túnel': t['nombre'], 'Ancho (m)': r2(t['ancho']), 'Alto (m)': r2(t['alto']),
        'Origen de la dimensión': t['origen_dim'],
    } for t in tuneles])


@st.fragment
def menu_11_geometria(df_frentes, df_fh, df_oc, df_res, radio_restriccion, radios_por_tipo):
    """Menú 15 — Geometría: el usuario elige ver el nivel en 2D o en 3D.
    2D = gráfico de siempre. 3D = túneles con su sección real (completos, sin cortes de
    avance: los avances se ven en los resultados) y las restricciones como esferas."""
    vista = st.radio("Vista de la geometría", ["2D", "3D"], horizontal=True, key="geo_vista_2d3d")
    if vista == "2D":
        fig_geo = graficar_geometria(
            df_frentes, df_fh, df_oc, df_res, radio_restriccion, radios_por_tipo
        )
        st.plotly_chart(fig_geo, width='stretch')
        return

    base = _base_tuneles_desde_df(df_frentes)
    nombres = [b['nombre'] for b in base]
    faltan = _tuneles_sin_dimension_en_seccion(base)

    if _usa_dimensiones_por_seccion():
        st.caption(
            "Las dimensiones de cada túnel salen de su **Sección** (Parámetros Generales: 'Usar Metros por Ciclo "
            "variable por Sección' está activado)."
        )
        if faltan:
            st.warning(
                "Estos túneles tienen una Sección que no viene como 'AnchoxAlto' (ej. 4.2x4.1): "
                + ", ".join(faltan) + ". Para ellos se usan las dimensiones que definas aquí abajo."
            )
    else:
        st.caption(
            "Define las dimensiones de la sección de los túneles (de 4 × 4 m hacia arriba). "
            "Se usan aquí y en los resultados. Si activas en Parámetros Generales 'Usar Metros por Ciclo "
            "variable por Sección', cada túnel tomará la dimensión de su Sección."
        )
    if faltan:
        _entradas_dimension_manual()

    c_sel, c_rest = st.columns([3, 1])
    elegidos = c_sel.multiselect("Túneles a mostrar", nombres, default=nombres, key="geo3d_sel")
    mostrar_rest = c_rest.checkbox("Mostrar restricciones (esferas de impacto)", value=True,
                                   key="geo3d_rest")
    if not elegidos:
        st.info("Selecciona al menos un túnel para verlo en 3D.")
        return

    base_sel = [b for b in base if b['nombre'] in elegidos]
    tuneles = _armar_tuneles_3d(base_sel, None, _dimensiones_3d(base_sel))
    _cfg_g = st.session_state.get('mwd_config')
    _cat_g = st.session_state.get('mwd_catalogo')
    _asig_g = {t: a for t, a in ((_cfg_g or {}).get('asignaciones') or {}).items()
               if a.get('modo') != 'Sin MWD' and t in elegidos}
    ver_geo = False
    if _asig_g and _cat_g is not None and len(_cat_g):
        ver_geo = st.checkbox("Colorear por geología MWD (rojo = muy mala, verde = muy buena)", value=True, key="geo3d_geol")
        if ver_geo:
            _sem = st.session_state.get('geo3d_geol_sem', 0)
            if any(a['modo'] == 'Aleatoria' for a in _asig_g.values()):
                if st.button("Volver a sortear la geología aleatoria", key="geo3d_geol_btn"):
                    st.session_state.geo3d_geol_sem = _sem + 1
                    st.rerun()
                st.caption("Los túneles en modo Aleatoria muestran un sorteo de ejemplo; la simulación usa un sorteo único, igual en todas las réplicas.")
            _perf = mwd_perfiles_vista(_asig_g, {t['nombre']: t['avance'] for t in tuneles}, _cat_g, _sem)
            for t in tuneles:
                t['geologia'] = _perf.get(str(t['nombre']))
    restricciones = [
        {'df': df_fh, 'name': 'FH (Frente Hundimiento)', 'color': 'red', 'symbol': 'circle',
         'radios': _radios_restriccion(df_fh, radio_restriccion, radios_por_tipo)},
        {'df': df_oc, 'name': 'OC (Obras Civiles)', 'color': 'blue', 'symbol': 'square',
         'radios': _radios_restriccion(df_oc, radio_restriccion, radios_por_tipo)},
        {'df': df_res, 'name': 'R (Restricción por Turno)', 'color': 'darkorange', 'symbol': 'diamond',
         'radios': _radios_restriccion(df_res, radio_restriccion, radios_por_tipo)},
    ] if mostrar_rest else None
    fig = graficar_tuneles_3d(
        tuneles, titulo="Geometría 3D del Nivel: Túneles y Restricciones" + (" + Geología MWD" if ver_geo else ""),
        modo_color=MODO_GEOLOGIA_3D if ver_geo else MODO_GEOMETRIA_3D, restricciones=restricciones,
        mostrar_pendiente=False, mostrar_lineas=False
    )
    st.plotly_chart(fig, width='stretch')
    with st.expander("Dimensiones de la sección por túnel", expanded=False):
        mostrar_tabla(_tabla_dimensiones(tuneles), width='stretch', hide_index=True)


@st.fragment
def menu_13_avance_3d(sim_seleccionada_global):
    """Menú 17 — Avance 3D: lo realmente construido en la simulación. Aquí SÍ se ve el
    corte por avance, con un color por avance (Avance 1, 2, 3...) o el total."""
    subtitulo_menu("Visualización 3D del Avance por Túnel")
    resultados = st.session_state.get('resultados', {})
    frentes_info = st.session_state.get('frentes_info', {})
    if not resultados or not frentes_info:
        return
    st.caption(
        "Lo construido en cada túnel (desde la boca), con su sección real en 3D. Cada tramo es un "
        "avance (un ciclo completo) con su propio color: Avance 1, 2, 3... El tramo gris translúcido "
        "es lo que falta para llegar al largo planificado. Las dimensiones son las de Parámetros Generales "
        "(Sección) o las definidas en Geometría."
    )

    if sim_seleccionada_global is None:
        estadistico = st.radio(
            "Avance a mostrar (resumen de todas las simulaciones)",
            ["Esperanza (promedio)", "P10", "P50", "P90", "Mínimo", "Máximo"],
            horizontal=True, key="avance3d_estadistico"
        )
        funciones = {
            "Esperanza (promedio)": np.mean, "P10": lambda a: np.percentile(a, 10),
            "P50": lambda a: np.percentile(a, 50), "P90": lambda a: np.percentile(a, 90),
            "Mínimo": np.min, "Máximo": np.max,
        }
        avances = {f: float(funciones[estadistico](v)) for f, v in resultados.items() if len(v)}
        etiqueta = estadistico
    else:
        avances = {f: float(v[min(sim_seleccionada_global - 1, len(v) - 1)])
                   for f, v in resultados.items() if len(v)}
        etiqueta = f"Simulación N° {sim_seleccionada_global}"

    base = _base_tuneles_desde_info(frentes_info)
    nombres = [b['nombre'] for b in base]

    c1, c2 = st.columns([2, 1])
    _log_g = st.session_state.get('mwd_log_geologia')
    _hay_geo = _log_g is not None and len(_log_g) > 0 and 'Tramo_Desde_m' in _log_g.columns
    _opc_modo = [MODO_POR_AVANCE_3D, MODO_TOTAL_3D] + ([MODO_GEOLOGIA_3D] if _hay_geo else [])
    modo = c1.radio("Colorear", _opc_modo, horizontal=True, key="av3d_modo")
    pendiente = c2.checkbox("Mostrar tramo pendiente (gris)", value=True, key="av3d_pend")
    lineas = c2.checkbox("Marcar el límite de cada avance", value=True, key="av3d_lin")
    elegidos = st.multiselect("Túneles a mostrar", nombres, default=nombres, key="av3d_sel")
    if not elegidos:
        st.info("Selecciona al menos un túnel para verlo en 3D.")
        return

    base_sel = [b for b in base if b['nombre'] in elegidos]
    tuneles = _armar_tuneles_3d(base_sel, avances, _dimensiones_3d(base_sel))
    if _hay_geo and str(modo).startswith("Geolog"):
        _sim_g = int(sim_seleccionada_global) if sim_seleccionada_global is not None else 1
        if _sim_g not in set(int(s) for s in _log_g['Simulacion'].unique()):
            _sim_g = int(_log_g['Simulacion'].min())
            st.caption(f"La geología sorteada solo se registra para las primeras {MWD_MAX_SIMS_LOG} simulaciones; "
                       f"se muestra la de la simulación {_sim_g}.")
        elif sim_seleccionada_global is None:
            st.caption("Con estadístico se dibuja la geología de la simulación 1 (la geología Aleatoria es la misma en todas las simulaciones).")
        _perf = mwd_perfiles_desde_log(_log_g, _sim_g)
        for t in tuneles:
            t['geologia'] = _perf.get(str(t['nombre']))
        st.caption("Colores: rojo = muy mala, naranja = mala, amarillo = regular, verde claro = buena, verde = muy buena, "
                   "azul grisáceo = túnel sin MWD. Gris translúcido = por construir. Si un avance cruza dos rocas, "
                   "su duración de perforación es el promedio ponderado por los metros de cada roca.")
    fig = graficar_tuneles_3d(
        tuneles, titulo=f"Avance 3D — {etiqueta}", modo_color=modo,
        mostrar_pendiente=pendiente, mostrar_lineas=lineas
    )
    st.plotly_chart(fig, width='stretch')

    resumen = []
    for t in tuneles:
        m = t['metros_ciclo']
        resumen.append({
            'Túnel': t['nombre'],
            'Avance mostrado (m)': r2(t['avance']),
            'Metros por avance (m)': r2(m),
            'N° de avances': int(math.ceil(t['avance'] / m - 1e-9)) if m > 1e-9 and t['avance'] > 0 else 0,
            'Sección (ancho × alto, m)': f"{t['ancho']:g} × {t['alto']:g}",
        })
    mostrar_tabla(pd.DataFrame(resumen), width='stretch', hide_index=True)


# ============================================================
# VISTA 3D DEL VIAJE DE LOS RECURSOS, POR AVANCE (Menú 17)
# ============================================================
# Dibuja, sobre los túneles (a escala real y translúcidos), el recorrido que hizo cada
# recurso para llegar al frente donde se ejecutó un avance. El recorrido es EXACTAMENTE el
# que calcula el motor ("viaje en U"):
#     posición actual del recurso -> boca (Xi,Yi) del túnel de origen
#                                 -> boca (Xi,Yi) del túnel de destino
#                                 -> posición actual del frente destino
# por lo que el largo de la línea dibujada coincide con 'Travel_Distance_m' de la traza.
# Cada viaje se registra con el N° de avance (Ciclo_Frente) del túnel de DESTINO.

ALTURA_RELATIVA_VIAJES_3D = 0.35   # fracción del alto del túnel a la que se dibuja el viaje
SEPARACION_Z_GRUPOS_VIAJES_3D = 0.3  # m de separación vertical entre grupos (evita solapes)
MAX_VIAJES_DIBUJADOS_3D = 2000     # tope de viajes dibujados a la vez (rendimiento)
OPCION_TODOS_3D = "Todos"
COLOR_POR_RECURSO_3D = "Recurso"
COLOR_POR_AVANCE_VIAJES_3D = "N° de avance"
COLOR_POR_VIAJE_3D = "N° de viaje"


@st.cache_data(show_spinner=False)
def preparar_viajes_3d(_df_traza, run_id, sim_id):
    """Viajes ('travel') de UNA simulación, con el tipo de recurso, ordenados por hora de
    inicio. Cacheada por run_id/sim_id igual que el resto de funciones que leen la traza."""
    requeridas = {
        'type', 'Simulation_ID', 'Recurso', 'Start', 'Finish', 'Ciclo_Frente',
        'Start_X_Front', 'Start_Y_Front', 'End_X_Front', 'End_Y_Front',
        'Resource_Origin_Tunnel', 'Resource_Destination_Tunnel',
        'Travel_Distance_m', 'Travel_Time_Actual', 'Travel_Speed_Used_m_h',
    }
    if _df_traza is None or _df_traza.empty or not requeridas.issubset(_df_traza.columns):
        return pd.DataFrame()
    df = _df_traza[(_df_traza['type'] == 'travel') & (_df_traza['Simulation_ID'] == sim_id)].copy()
    if df.empty:
        return df
    df['Recurso_Tipo'] = df['Recurso'].apply(_tipo_recurso_instancia)
    return df.sort_values('Start').reset_index(drop=True)


def _ruta_viaje_3d(fila, frentes_info):
    """Vértices (x, y) del viaje en U de una fila 'travel' de la traza."""
    pts = [(float(fila['Start_X_Front']), float(fila['Start_Y_Front']))]
    for nombre in (fila['Resource_Origin_Tunnel'], fila['Resource_Destination_Tunnel']):
        info = frentes_info.get(nombre)
        if info is not None:
            pts.append((float(info['xi']), float(info['yi'])))  # boca del túnel
    pts.append((float(fila['End_X_Front']), float(fila['End_Y_Front'])))
    return pts


def _largo_ruta(pts):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts[:-1], pts[1:]))


def _clave_recurso_3d(nombre):
    """Orden natural de instancias: por tipo y luego por número ('Grúa_0', 'Grúa_1', ...)."""
    sufijo = str(nombre).rsplit('_', 1)[-1]
    return (_tipo_recurso_instancia(nombre), int(sufijo) if sufijo.isdigit() else -1)


def _colores_viajes_3d(n):
    """n colores distintos; con muchos viajes usa un degradé (Viridis) para que se lea el orden."""
    if n <= len(PALETA_AVANCES_3D):
        return PALETA_AVANCES_3D[:n]
    try:
        from plotly.colors import sample_colorscale
        return sample_colorscale('Viridis', [i / max(n - 1, 1) for i in range(n)])
    except Exception:
        return [PALETA_AVANCES_3D[i % len(PALETA_AVANCES_3D)] for i in range(n)]


def agregar_viajes_3d(fig, df_viajes, frentes_info, z_base, agrupar_por=COLOR_POR_VIAJE_3D,
                      mostrar_lineas=True):
    """Agrega a 'fig' los viajes de los recursos elegidos: círculo = de dónde sale, rombo = dónde
    llega, y (opcional) la línea del recorrido en U. Cada RECURSO va a su propia altura (así no se
    pisan) y la leyenda cuenta recursos / viajes / avances, no 'N viajes'. Además marca con un
    rombo negro la posición actual de cada recurso (donde terminó su último viaje mostrado).
    Requiere la columna '_n_viaje' (N° de viaje del recurso, en orden cronológico)."""
    if df_viajes.empty:
        return fig
    df = df_viajes.copy()
    recursos = sorted(df['Recurso'].unique(), key=_clave_recurso_3d)
    z_recurso = {r: z_base + SEPARACION_Z_GRUPOS_VIAJES_3D * i for i, r in enumerate(recursos)}

    if agrupar_por == COLOR_POR_AVANCE_VIAJES_3D:
        df['_grupo'] = df['Ciclo_Frente'].apply(lambda a: f"Avance {int(a)}" if pd.notna(a) else "Avance ?")
        df['_orden'] = df['Ciclo_Frente'].fillna(10 ** 9)
        orden_grupos = list(df.sort_values('_orden')['_grupo'].unique())
        titulo_leyenda = 'Avance'
    elif agrupar_por == COLOR_POR_RECURSO_3D:
        df['_grupo'] = df['Recurso']
        orden_grupos = recursos
        titulo_leyenda = 'Recurso'
    else:  # COLOR_POR_VIAJE_3D
        df['_grupo'] = df['_n_viaje'].apply(lambda n: f"Viaje {int(n)}")
        orden_grupos = list(df.sort_values('_n_viaje')['_grupo'].unique())
        titulo_leyenda = 'Viaje del recurso'
    colores = _colores_viajes_3d(len(orden_grupos))

    for k, grupo in enumerate(orden_grupos):
        sub = df[df['_grupo'] == grupo]
        color = colores[k]
        xs, ys, zs, tx = [], [], [], []
        ix, iy, iz, itx, fx, fy, fz, ftx = [], [], [], [], [], [], [], []
        for _, f in sub.iterrows():
            z = z_recurso[f['Recurso']]
            pts = _ruta_viaje_3d(f, frentes_info)
            av = int(f['Ciclo_Frente']) if pd.notna(f['Ciclo_Frente']) else '?'
            texto = (
                f"<b>{f['Recurso']}</b> — viaje N° {int(f['_n_viaje'])}<br>"
                f"Para el Avance {av} de {f['Resource_Destination_Tunnel']}<br>"
                f"{f['Resource_Origin_Tunnel']} → {f['Resource_Destination_Tunnel']}<br>"
                f"Distancia: {float(f['Travel_Distance_m']):.1f} m<br>"
                f"Tiempo: {float(f['Travel_Time_Actual']):.3f} h "
                f"({float(f['Travel_Time_Actual']) * 60:.1f} min)<br>"
                f"Velocidad: {float(f['Travel_Speed_Used_m_h']) / 1000:.1f} km/h<br>"
                f"Hora: {float(f['Start']):.2f} → {float(f['Finish']):.2f} h"
            )
            for px, py in pts:
                xs.append(px); ys.append(py); zs.append(z); tx.append(texto)
            xs.append(None); ys.append(None); zs.append(None); tx.append(None)
            ix.append(pts[0][0]); iy.append(pts[0][1]); iz.append(z); itx.append(texto)
            fx.append(pts[-1][0]); fy.append(pts[-1][1]); fz.append(z); ftx.append(texto)

        tam = 5 if mostrar_lineas else 7  # sin líneas, los puntos son lo único que se ve
        if mostrar_lineas:
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs, mode='lines+markers', name=str(grupo), legendgroup=str(grupo),
                line=dict(color=color, width=5), marker=dict(size=2, color=color),
                hovertext=tx, hoverinfo='text'))
        fig.add_trace(go.Scatter3d(
            x=ix, y=iy, z=iz, mode='markers', legendgroup=str(grupo),
            name=str(grupo), showlegend=not mostrar_lineas,
            marker=dict(size=tam, color=color, symbol='circle'),
            hovertext=itx, hoverinfo='text'))
        fig.add_trace(go.Scatter3d(
            x=fx, y=fy, z=fz, mode='markers', legendgroup=str(grupo), showlegend=False,
            marker=dict(size=tam + 1, color=color, symbol='diamond'),
            hovertext=ftx, hoverinfo='text'))

    # Posición actual de cada recurso = donde terminó su último viaje mostrado
    ult = df.sort_values('Start').groupby('Recurso').tail(1)
    etiquetas = [] if len(ult) > 8 else list(ult['Recurso'])
    fig.add_trace(go.Scatter3d(
        x=ult['End_X_Front'].astype(float), y=ult['End_Y_Front'].astype(float),
        z=[z_recurso[r] for r in ult['Recurso']],
        mode='markers+text' if etiquetas else 'markers', name='Posición actual',
        text=etiquetas if etiquetas else None, textposition='top center',
        marker=dict(size=10, color='#111111', symbol='diamond',
                    line=dict(color='#ffffff', width=1)),
        hovertext=[f"<b>{r}</b><br>Posición actual: {d}<br>(tras el viaje N° {int(n)})"
                   for r, d, n in zip(ult['Recurso'], ult['Resource_Destination_Tunnel'], ult['_n_viaje'])],
        hoverinfo='text'))
    fig.update_layout(legend_title_text=titulo_leyenda)
    return fig


def agregar_viajes_3d_por_paso(fig, df_viajes, frentes_info, z_base, agrupar_por=COLOR_POR_RECURSO_3D):
    """Viajes por PASO: el paso N muestra SOLO el viaje N° N de cada recurso elegido (no los
    anteriores): circulo = de donde sale, rombo = donde llega, y un rombo negro = donde esta cada
    recurso tras ese paso (si ya no tiene mas viajes, se queda en su ultima posicion, asi con 3
    Jumbos siempre se ven los 3). Cada recurso va a su propia altura.
    El control de pasos es el DESLIZADOR NATIVO de Plotly: se mueve en el navegador, sin volver a
    ejecutar Streamlit, por eso el 3D no se resetea ni pierde la camara al moverlo.
    Requiere la columna '_n_viaje' (N° de viaje del recurso, cronologico)."""
    if df_viajes.empty:
        return fig
    df = df_viajes.copy()
    recursos = sorted(df['Recurso'].unique(), key=_clave_recurso_3d)
    z_recurso = {r: z_base + SEPARACION_Z_GRUPOS_VIAJES_3D * i for i, r in enumerate(recursos)}
    max_n = int(df['_n_viaje'].max())

    if agrupar_por == COLOR_POR_AVANCE_VIAJES_3D:
        df['_grupo'] = df['Ciclo_Frente'].apply(lambda a: f"Avance {int(a)}" if pd.notna(a) else "Avance ?")
        orden = list(df.assign(_o=df['Ciclo_Frente'].fillna(10 ** 9)).sort_values('_o')['_grupo'].unique())
        titulo_leyenda = 'Avance atendido'
    else:
        df['_grupo'] = df['Recurso']
        orden = recursos
        titulo_leyenda = 'Recurso'
    color_de = dict(zip(orden, _colores_viajes_3d(len(orden))))

    # Leyenda fija (siempre visible): un punto por grupo de color
    for g in orden:
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None], mode='markers', name=str(g), legendgroup=str(g),
            marker=dict(size=7, color=color_de[g]), hoverinfo='skip'))
    n_fijas = len(fig.data)

    por_recurso = {r: df[df['Recurso'] == r].sort_values('_n_viaje').to_dict('records') for r in recursos}

    def _texto(f):
        av = int(f['Ciclo_Frente']) if pd.notna(f['Ciclo_Frente']) else '?'
        return (
            f"<b>{f['Recurso']}</b> — viaje N° {int(f['_n_viaje'])}<br>"
            f"Para el Avance {av} de {f['Resource_Destination_Tunnel']}<br>"
            f"{f['Resource_Origin_Tunnel']} → {f['Resource_Destination_Tunnel']}<br>"
            f"Distancia: {float(f['Travel_Distance_m']):.1f} m<br>"
            f"Tiempo: {float(f['Travel_Time_Actual']):.3f} h ({float(f['Travel_Time_Actual']) * 60:.1f} min)<br>"
            f"Velocidad: {float(f['Travel_Speed_Used_m_h']) / 1000:.1f} km/h<br>"
            f"Hora: {float(f['Start']):.2f} → {float(f['Finish']):.2f} h"
        )

    etiquetas_pasos = []
    for n in range(1, max_n + 1):
        ox, oy, oz, ot, oc = [], [], [], [], []
        dx, dy, dz, dt, dc = [], [], [], [], []
        ax, ay, az, at, an = [], [], [], [], []
        con_viaje = 0
        for r in recursos:
            viajes = por_recurso[r]
            if n <= len(viajes):
                f = viajes[n - 1]
                con_viaje += 1
                pts = _ruta_viaje_3d(f, frentes_info)
                z = z_recurso[r]
                txt = _texto(f)
                col = color_de[f['_grupo']]
                ox.append(pts[0][0]); oy.append(pts[0][1]); oz.append(z); ot.append(txt); oc.append(col)
                dx.append(pts[-1][0]); dy.append(pts[-1][1]); dz.append(z); dt.append(txt); dc.append(col)
                ax.append(float(f['End_X_Front'])); ay.append(float(f['End_Y_Front'])); az.append(z)
                at.append(f"<b>{r}</b><br>Posición actual: {f['Resource_Destination_Tunnel']}<br>(tras el viaje N° {n})")
                an.append(r)
            elif viajes:
                f = viajes[-1]   # ya no tuvo mas viajes: sigue donde termino el ultimo
                z = z_recurso[r]
                ax.append(float(f['End_X_Front'])); ay.append(float(f['End_Y_Front'])); az.append(z)
                at.append(f"<b>{r}</b><br>Posición actual: {f['Resource_Destination_Tunnel']}<br>"
                          f"(sin más viajes; el último fue el N° {len(viajes)})")
                an.append(r)
        visible = (n == 1)
        con_nombre = len(recursos) <= 8
        fig.add_trace(go.Scatter3d(   # donde esta cada recurso (debajo, un poco mas grande)
            x=ax, y=ay, z=az, mode='markers+text' if con_nombre else 'markers', name='Posición actual',
            text=an if con_nombre else None, textposition='top center', showlegend=False,
            marker=dict(size=11, color='#111111', symbol='diamond', line=dict(color='#ffffff', width=1)),
            hovertext=at, hoverinfo='text', visible=visible))
        fig.add_trace(go.Scatter3d(
            x=ox, y=oy, z=oz, mode='markers', name='Sale de', showlegend=False,
            marker=dict(size=6, color=oc, symbol='circle'), hovertext=ot, hoverinfo='text', visible=visible))
        fig.add_trace(go.Scatter3d(
            x=dx, y=dy, z=dz, mode='markers', name='Llega a', showlegend=False,
            marker=dict(size=7, color=dc, symbol='diamond'), hovertext=dt, hoverinfo='text', visible=visible))
        if len(recursos) == 1 and con_viaje:
            f = por_recurso[recursos[0]][n - 1]
            av = f"Avance {int(f['Ciclo_Frente'])}" if pd.notna(f['Ciclo_Frente']) else "Avance ?"
            etiquetas_pasos.append(f"{n} → {av} de {f['Resource_Destination_Tunnel']}")
        else:
            etiquetas_pasos.append(f"{n} ({con_viaje} de {len(recursos)} recursos viajan)")

    fig.update_layout(legend_title_text=titulo_leyenda)
    if max_n > 1:
        pasos = []
        for n in range(1, max_n + 1):
            vis = [True] * n_fijas + [False] * (3 * max_n)
            vis[n_fijas + 3 * (n - 1): n_fijas + 3 * n] = [True, True, True]
            pasos.append(dict(method='restyle', args=[{'visible': vis}], label=etiquetas_pasos[n - 1]))
        fig.update_layout(
            sliders=[dict(
                active=0, steps=pasos, x=0.02, len=0.96, y=0, yanchor='top', pad=dict(t=25, b=5),
                currentvalue=dict(prefix='Viaje N° ', xanchor='left',
                                  font=dict(size=16, color='#6b6f7e')),
                font=dict(size=1, color='rgba(0,0,0,0)'), ticklen=0, minorticklen=0,
            )],
            margin=dict(b=110)
        )
    return fig


def _fijar_rangos_escena_3d(fig, margen=0.03):
    """Fija el rango de los 3 ejes con TODOS los datos de la figura (tambien los pasos ocultos del
    deslizador). Asi, al mostrar/ocultar viajes, Plotly no recalcula ejes ni proporciones y la vista
    no 'salta'."""
    mins, maxs = [np.inf] * 3, [-np.inf] * 3
    for tr in fig.data:
        for i, eje in enumerate(('x', 'y', 'z')):
            v = getattr(tr, eje, None)
            if v is None:
                continue
            a = np.array([x for x in v if x is not None], dtype=float)
            a = a[np.isfinite(a)]
            if a.size:
                mins[i], maxs[i] = min(mins[i], a.min()), max(maxs[i], a.max())
    if not all(np.isfinite(mins + maxs)):
        return fig
    rangos = {}
    for i, eje in enumerate(('xaxis', 'yaxis', 'zaxis')):
        pad = (maxs[i] - mins[i]) * margen or 1.0
        rangos[eje] = dict(range=[mins[i] - pad, maxs[i] + pad])
    fig.update_layout(scene=rangos)
    return fig


# JS que corre DENTRO de la figura (en el navegador): recuerda la camara 3D (zoom/giro) y la vuelve a
# aplicar cada vez que el deslizador cambia el viaje, y tambien cuando Streamlit redibuja la figura
# (cambio de recurso/color), guardandola en sessionStorage. {plot_id} lo reemplaza Plotly.
_JS_CAMARA_PERSISTENTE_3D = """
var gd = document.getElementById('{plot_id}');
var KEY = 'viajes3d_camara_v1', cam = null, aplicando = false;
function leer(){ try { return JSON.parse(window.sessionStorage.getItem(KEY)); } catch(e){ return null; } }
function guardar(c){ try { window.sessionStorage.setItem(KEY, JSON.stringify(c)); } catch(e){} }
function actual(){ try { return JSON.parse(JSON.stringify(gd._fullLayout.scene._scene.getCamera())); } catch(e){ return null; } }
function aplicar(){
  if(!cam) return;
  aplicando = true;
  Plotly.relayout(gd, {'scene.camera': cam}).then(function(){ aplicando = false; }, function(){ aplicando = false; });
}
cam = leer();
if(cam){ aplicar(); }
gd.on('plotly_relayout', function(e){
  if(aplicando) return;
  if(e && e['scene.camera']){ cam = JSON.parse(JSON.stringify(e['scene.camera'])); guardar(cam); }
});
gd.on('plotly_sliderchange', function(){ var c = actual(); if(c){ cam = c; guardar(cam); } });
gd.on('plotly_restyle', function(){ aplicar(); });
"""


def mostrar_figura_3d_camara_persistente(fig, clave=None):
    """Muestra la figura 3D como HTML de Plotly (en vez de st.plotly_chart) para controlar la camara:
    mover el deslizador de viajes no la reinicia. Incluye plotly.js dentro del HTML (funciona sin
    internet). El HTML no cambia si los datos no cambian, asi que Streamlit no lo vuelve a montar."""
    fig.update_layout(
        template='plotly_white', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Source Sans Pro, sans-serif', color='#6b6f7e'),
        uirevision='viajes3d_camara'
    )
    alto = int(fig.layout.height or 700)
    html = fig.to_html(
        full_html=True, include_plotlyjs=True, post_script=_JS_CAMARA_PERSISTENTE_3D,
        default_height=f"{alto}px", default_width='100%',
        config={'responsive': True, 'displaylogo': False}
    )
    if hasattr(st, 'iframe'):
        st.iframe(html, height=alto + 10)
    else:  # versiones antiguas de Streamlit
        import streamlit.components.v1 as components
        components.html(html, height=alto + 10, scrolling=False)


@st.fragment
def menu_13b_viajes_3d(sim_seleccionada_global):
    """Menú 17 — Viaje de los recursos en 3D, por avance. Es una vista de UNA simulación
    (la traza de viajes es por simulación, no tiene sentido promediarla). Se elige uno o más
    RECURSOS y se ve su recorrido viaje a viaje (con el avance que atendió cada uno)."""
    subtitulo_menu("Viaje de los Recursos en 3D por Avance")
    resultados = st.session_state.get('resultados', {})
    frentes_info = st.session_state.get('frentes_info', {})
    df_traza = st.session_state.get('df_traza_eventos_cache')
    run_id = st.session_state.get('run_id')
    if not resultados or not frentes_info or df_traza is None or df_traza.empty:
        return
    st.caption(
        "Elige un recurso y mira por dónde se movió: cada viaje lleva al recurso a un túnel para "
        "atender un avance (círculo = de dónde sale, rombo = dónde llega, rombo negro = dónde "
        "está al final). Con el deslizador bajo el gráfico avanzas viaje a viaje: en cada paso ves SOLO "
        "el viaje elegido de cada recurso (no los anteriores) y dónde queda cada uno. "
        "Pasa el mouse sobre un punto para ver origen → destino, avance, distancia, tiempo y "
        "velocidad. El motor calcula el viaje en U: posición actual → boca del túnel de origen → "
        "boca del túnel de destino → frente."
    )

    n_sims = st.session_state.get('n_simulaciones_ejecutadas', 1)
    if sim_seleccionada_global is None:
        sim_id = st.selectbox(
            "Simulación a mostrar", list(range(1, n_sims + 1)), key="viajes3d_sim",
            help="Los viajes son de una simulación puntual (no se promedian). "
                 "Si eliges 'Simulación específica' arriba, se usa esa."
        )
    else:
        sim_id = sim_seleccionada_global
        st.caption(f"Mostrando la **simulación N° {sim_id}** (la elegida arriba).")

    df_viajes = preparar_viajes_3d(df_traza, run_id, int(sim_id))
    if df_viajes.empty:
        st.info(
            "En esta simulación no hubo viajes de recursos entre túneles (o la traza no trae "
            "los campos necesarios). Los viajes ocurren cuando un recurso debe moverse de un "
            "túnel a otro para atender un avance."
        )
        return
    df_viajes = df_viajes.copy()
    # N° de viaje de cada recurso, en orden cronológico (df_viajes ya viene ordenado por 'Start')
    df_viajes['_n_viaje'] = df_viajes.groupby('Recurso').cumcount() + 1

    # ---- 1) Qué recurso(s) ----
    recursos_disp = sorted(df_viajes['Recurso'].unique(), key=_clave_recurso_3d)
    c1, c2 = st.columns([3, 1])
    todos = c2.checkbox("Todos los recursos", value=False, key=f"viajes3d_todos_{sim_id}")
    sel = c1.multiselect(
        "Recurso a mostrar", recursos_disp, default=recursos_disp[:1],
        key=f"viajes3d_recursos_{sim_id}", disabled=todos,
        help="Cada recurso es una instancia concreta (p. ej. Grúa_0). Puedes elegir varios.")
    recursos_sel = recursos_disp if todos else sel
    if not recursos_sel:
        st.info("Elige al menos un recurso para ver sus viajes.")
        return
    df_r = df_viajes[df_viajes['Recurso'].isin(recursos_sel)]

    # El control de viajes es el deslizador nativo de Plotly (dentro de la figura): moverlo no vuelve
    # a ejecutar Streamlit, asi el 3D no se resetea ni pierde la camara.
    agrupar_por = st.radio(
        "Colorear por", [COLOR_POR_RECURSO_3D, COLOR_POR_AVANCE_VIAJES_3D],
        horizontal=True, key="viajes3d_color_v3",
        help="'Recurso': un color por recurso. 'N° de avance': un color por avance atendido.")

    df_sel = df_r
    total_sel = len(df_sel)
    if total_sel > MAX_VIAJES_DIBUJADOS_3D:
        st.warning(
            f"Hay {total_sel} viajes con esta selección; para mantener el gráfico fluido se "
            f"dibujan los primeros {MAX_VIAJES_DIBUJADOS_3D} (por hora). Elige menos recursos."
        )
        df_sel = df_sel.sort_values('Start').head(MAX_VIAJES_DIBUJADOS_3D)

    base = _base_tuneles_desde_info(frentes_info)
    avances_sim = {f: float(v[min(int(sim_id) - 1, len(v) - 1)]) for f, v in resultados.items() if len(v)}
    tuneles = _armar_tuneles_3d(base, avances_sim, _dimensiones_3d(base))
    alto_prom = float(np.mean([t['alto'] for t in tuneles])) if tuneles else 4.0
    quien = recursos_sel[0] if len(recursos_sel) == 1 else f"{len(recursos_sel)} recursos"
    titulo = f"Viajes 3D — Simulación N° {sim_id} — {quien}"

    fig = graficar_tuneles_3d(tuneles, titulo=titulo, modo_color=MODO_TOTAL_3D,
                              mostrar_pendiente=True, mostrar_lineas=False)
    fig.update_traces(opacity=0.18, showlegend=False, selector=dict(type='mesh3d'))
    agregar_viajes_3d_por_paso(fig, df_sel, frentes_info, ALTURA_RELATIVA_VIAJES_3D * alto_prom, agrupar_por)
    _fijar_rangos_escena_3d(fig)
    mostrar_figura_3d_camara_persistente(fig)
    st.caption(
        f"Mueve el deslizador de abajo para ver cada viaje. La tabla de abajo resume los {len(df_sel)} viaje(s) de "
        f"{df_sel['Recurso'].nunique()} recurso(s). Ojo: solo "
        "se registra un viaje cuando el recurso tiene que cambiar de túnel; si ya estaba en el "
        "túnel del avance, no hay viaje y se queda donde estaba."
    )

    # ---- Resumen numérico de lo que se está viendo (por recurso) ----
    resumen = df_sel.groupby('Recurso').agg(
        Viajes=('Recurso', 'count'),
        Distancia_Total=('Travel_Distance_m', 'sum'),
        Distancia_Promedio=('Travel_Distance_m', 'mean'),
        Tiempo_Total=('Travel_Time_Actual', 'sum'),
        Velocidad_Prom=('Travel_Speed_Used_m_h', 'mean'),
    ).reset_index()
    resumen = resumen.sort_values('Recurso', key=lambda s: s.map(_clave_recurso_3d))
    mostrar_tabla(pd.DataFrame({
        'Recurso': resumen['Recurso'],
        'N° de viajes': resumen['Viajes'],
        'Distancia total (m)': resumen['Distancia_Total'].round(2),
        'Distancia promedio (m)': resumen['Distancia_Promedio'].round(2),
        'Tiempo total de viaje (h)': resumen['Tiempo_Total'].round(3),
        'Velocidad promedio (km/h)': (resumen['Velocidad_Prom'] / 1000).round(2),
    }), width='stretch', hide_index=True)

    with st.expander("Detalle de cada viaje dibujado", expanded=False):
        detalle = df_sel[[
            'Recurso', '_n_viaje', 'Ciclo_Frente', 'Resource_Origin_Tunnel',
            'Resource_Destination_Tunnel', 'Travel_Distance_m', 'Travel_Time_Actual',
            'Travel_Speed_Used_m_h', 'Start', 'Finish'
        ]].copy()
        detalle['Travel_Speed_Used_m_h'] = (detalle['Travel_Speed_Used_m_h'] / 1000).round(2)
        detalle.columns = ['Recurso', 'N° viaje', 'Avance atendido', 'Túnel origen', 'Túnel destino',
                           'Distancia (m)', 'Tiempo viaje (h)', 'Velocidad (km/h)', 'Inicio (h)', 'Fin (h)']
        mostrar_tabla(detalle, width='stretch', hide_index=True)


# ============================================================
# GENERACIÓN DE OUTPUTS FORMATO DAVID MONTENEGRO
# ============================================================

def _build_actividad_to_secuencia(df_actividades):
    """Mapea nombre de actividad → número de secuencia."""
    mapping = {}
    for _, row in df_actividades.dropna(subset=['Actividad']).iterrows():
        mapping[str(row['Actividad'])] = int(row['Secuencia'])
    return mapping


def generar_advances_total_scenary(resultados, frentes_info, nivel_seleccionado):
    """
    Formato: David Montenegro Advances_total_scenary.xlsx
    Una hoja Sheet1, filas = frentes, cols = scenary 1..N
    """
    rows = []
    for i, (frente, avances_lista) in enumerate(resultados.items()):
        row = {'Index': i + 1, 'Frentes': frente, 'Nivel': nivel_seleccionado}
        for j, avance in enumerate(avances_lista):
            row[f'scenary {j + 1}'] = round(float(avance), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def generar_simulation_front_actual(traza_eventos, frentes_info, df_frentes_nivel, n_simulaciones):
    """
    Formato: David Montenegro Simulation_Front_Actual.xlsx
    Una hoja por escenario ('scenary 1'...'scenary N').
    Columnas: Nivel, Index, Frentes, Sección, Xi, Yi, Xf, Yf, Dirección, LastShift, State, shift_State, FH_Distance, OOCC_Distance
    Xi/Yi = posición final alcanzada en ese escenario.
    """
    if not traza_eventos:
        return {}
    df_traza = pd.DataFrame(traza_eventos)
    sheets = {}
    for sim_id in range(1, n_simulaciones + 1):
        df_sim = df_traza[df_traza['Simulation_ID'] == sim_id]
        rows = []
        for i, (frente, info) in enumerate(frentes_info.items()):
            # Buscar última posición registrada para este frente en este escenario
            df_f = df_sim[df_sim['Resource_Origin_Tunnel'] == frente]
            if not df_f.empty:
                last_row = df_f.sort_values('Finish').iloc[-1]
                xi_actual = last_row['End_X_Front']
                yi_actual = last_row['End_Y_Front']
                last_shift = last_row['Finish']
            else:
                xi_actual = info['xi']
                yi_actual = info['yi']
                last_shift = 0

            avance_total = min(
                calcular_distancia(info['xi'], info['yi'], xi_actual, yi_actual),
                info['distancia']
            )
            state = 'False' if avance_total >= info['distancia'] * 0.999 else 'True'

            # Datos de la fila original de frentes
            fila_original = df_frentes_nivel[df_frentes_nivel['Frentes'] == frente]
            seccion = str(fila_original[columna_seccion(df_frentes_nivel)].iloc[0]) if (columna_seccion(df_frentes_nivel) and not fila_original.empty) else ''
            direccion = float(fila_original['Dirección'].iloc[0]) if ('Dirección' in fila_original.columns and not fila_original.empty) else 0.0

            rows.append({
                'Nivel': info.get('nivel', ''),
                'Index': i + 1,
                'Frentes': frente,
                'Sección': seccion,
                'Xi': round(xi_actual, 4),
                'Yi': round(yi_actual, 4),
                'Xf': round(info['xf'], 4),
                'Yf': round(info['yf'], 4),
                'Dirección': direccion,
                'LastShift': round(last_shift, 2),
                'State': state,
                'shift_State': 'True',
                'FH_Distance': '',
                'OOCC_Distance': ''
            })
        sheets[f'scenary {sim_id}'] = pd.DataFrame(rows)
    return sheets


def generar_simulation_program_por_turno(traza_eventos, df_actividades, frentes_info,
                                          numero_turnos, duracion_turno, n_simulaciones):
    """
    Genera 3 estructuras con formato de David Montenegro (una hoja por escenario):
    - Program_Sequences: {scenary N: DataFrame} — secuencias ejecutadas por turno por frente
    - Program_Activities: {scenary N: DataFrame} — actividades ejecutadas por turno por frente
    - Program_Advances: {scenary N: DataFrame} — avances registrados por turno por frente
    """
    if not traza_eventos:
        return {}, {}, {}

    df_traza = pd.DataFrame(traza_eventos)
    act_to_seq = _build_actividad_to_secuencia(df_actividades)

    sheets_seq, sheets_act, sheets_adv = {}, {}, {}
    turnos_cols = [f'T{t}' for t in range(1, numero_turnos + 1)]
    frentes_nombres = list(frentes_info.keys())

    for sim_id in range(1, n_simulaciones + 1):
        df_sim = df_traza[df_traza['Simulation_ID'] == sim_id]
        rows_seq, rows_act, rows_adv = [], [], []

        for i, frente in enumerate(frentes_nombres):
            df_f = df_sim[
                (df_sim['Resource_Origin_Tunnel'] == frente) &
                (df_sim['type'] == 'activity')
            ].sort_values('Start')

            # Detectar fin de ciclos para avances
            ciclos_por_fin = {}
            if 'Cycle' in df_f.columns:
                for cycle_id, grp in df_f.groupby('Cycle'):
                    t_fin = grp['Finish'].max()
                    adv_m = grp['Cycle_Advance_Length_m'].max() if 'Cycle_Advance_Length_m' in grp.columns else 0
                    last = grp.sort_values('Finish').iloc[-1]
                    ciclos_por_fin[cycle_id] = {
                        'finish': t_fin,
                        'metros': adv_m,
                        'x': last['End_X_Front'],
                        'y': last['End_Y_Front']
                    }

            row_seq = {'Index': i + 1, 'Frentes': frente}
            row_act = {'Index': i + 1, 'Frentes': frente}
            row_adv = {'Index': i + 1, 'Frentes': frente}

            for t in range(1, numero_turnos + 1):
                t_start = (t - 1) * duracion_turno
                t_end = t * duracion_turno
                col = f'T{t}'

                df_shift = df_f[
                    (df_f['Start'] >= t_start) & (df_f['Start'] < t_end)
                ]

                if df_shift.empty:
                    row_seq[col] = ''
                    row_act[col] = ''
                else:
                    acts = df_shift['Actividad'].dropna().tolist()
                    row_act[col] = '-'.join(str(a) for a in acts)
                    seqs = [str(act_to_seq.get(str(a), '?')) for a in acts]
                    row_seq[col] = '-'.join(seqs)

                # Avances: ciclos que terminaron en este turno
                adv_parts = []
                for cid, cinfo in ciclos_por_fin.items():
                    if t_start < cinfo['finish'] <= t_end and cinfo['metros'] > 0:
                        adv_parts.append(
                            f"meters: {round(cinfo['metros'], 2)} "
                            f"X: {round(cinfo['x'], 2)} "
                            f"Y: {round(cinfo['y'], 2)}"
                        )
                row_adv[col] = ' | '.join(adv_parts)

            rows_seq.append(row_seq)
            rows_act.append(row_act)
            rows_adv.append(row_adv)

        key = f'scenary {sim_id}'
        sheets_seq[key] = pd.DataFrame(rows_seq)
        sheets_act[key] = pd.DataFrame(rows_act)
        sheets_adv[key] = pd.DataFrame(rows_adv)

    return sheets_seq, sheets_act, sheets_adv


def generar_simulation_time_program(traza_eventos, recursos_config, numero_turnos,
                                     duracion_turno, n_simulaciones, ventanas_por_recurso=None):
    """
    Formato: David Montenegro Simulation_Time_Program.xlsx
    Una hoja por escenario. Filas = recursos, cols = T1..TN.
    Valor = tiempo disponible restante en ese turno (capacidad del turno - tiempo_usado).
    Si hay un plan de orquestacion de flota por turno, la capacidad de cada turno
    refleja la cantidad de unidades realmente activas en ese turno (no la cantidad base fija).
    """
    if not traza_eventos:
        return {}
    df_traza = pd.DataFrame(traza_eventos)
    sheets = {}
    recursos_lista = [r for r, cfg in recursos_config.items() if cfg['cantidad'] > 0]
    ventanas_por_recurso = ventanas_por_recurso or {}

    for sim_id in range(1, n_simulaciones + 1):
        df_sim = df_traza[df_traza['Simulation_ID'] == sim_id]
        rows = []
        for recurso in recursos_lista:
            df_rec = df_sim[
                df_sim['Recurso'].astype(str).str.startswith(recurso) &
                (df_sim['type'] == 'activity')
            ]
            row = {'Recurso': recurso}
            ventanas_recurso = ventanas_por_recurso.get(recurso)
            for t in range(1, numero_turnos + 1):
                t_start = (t - 1) * duracion_turno
                t_end = t * duracion_turno
                df_shift = df_rec[
                    (df_rec['Start'] >= t_start) & (df_rec['Start'] < t_end)
                ]
                tiempo_usado = df_shift['Duration'].sum()
                if ventanas_recurso:
                    unidades_activas_turno = sum(
                        1 for v in ventanas_recurso
                        if v['activa_desde'] < t_end and v['activa_hasta'] > t_start
                    )
                    capacidad_total = duracion_turno * unidades_activas_turno
                else:
                    capacidad_total = duracion_turno * recursos_config[recurso]['cantidad']
                restante = max(0.0, capacidad_total - tiempo_usado)
                row[f'T{t}'] = round(restante, 6)
            rows.append(row)
        sheets[f'scenary {sim_id}'] = pd.DataFrame(rows)
    return sheets


def escribir_excel_multihoja(filepath, sheets_dict):
    """Escribe un dict {sheet_name: DataFrame} como Excel multi-hoja."""
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        for sheet_name, df in sheets_dict.items():
            df.to_excel(writer, sheet_name=str(sheet_name)[:31], index=False)


def guardar_outputs_en_ruta(output_path, traza_eventos, resultados, frentes_info,
                             df_frentes_nivel, df_actividades, recursos_config,
                             numero_turnos, duracion_turno, n_simulaciones,
                             nivel_seleccionado):
    """
    Guarda los 6 archivos en formato David Montenegro + reporte combinado en output_path.
    Retorna lista de archivos guardados o lanza excepción.
    """
    os.makedirs(output_path, exist_ok=True)
    guardados = []

    # 1. Advances_total_scenary.xlsx
    df_advances = generar_advances_total_scenary(resultados, frentes_info, nivel_seleccionado)
    p = os.path.join(output_path, 'Advances_total_scenary.xlsx')
    df_advances.to_excel(p, index=False)
    guardados.append(p)

    # 2. Simulation_Front_Actual.xlsx
    sheets_front = generar_simulation_front_actual(
        traza_eventos, frentes_info, df_frentes_nivel, n_simulaciones)
    p = os.path.join(output_path, 'Simulation_Front_Actual.xlsx')
    escribir_excel_multihoja(p, sheets_front)
    guardados.append(p)

    # 3-5. Sequences, Activities, Advances (por turno)
    sheets_seq, sheets_act, sheets_adv = generar_simulation_program_por_turno(
        traza_eventos, df_actividades, frentes_info,
        numero_turnos, duracion_turno, n_simulaciones
    )
    for fname, sheets in [
        ('Simulation_Program_Sequences.xlsx', sheets_seq),
        ('Simulation_Program_Activities.xlsx', sheets_act),
        ('Simulation_Program_Advances.xlsx', sheets_adv),
    ]:
        p = os.path.join(output_path, fname)
        escribir_excel_multihoja(p, sheets)
        guardados.append(p)

    # 6. Simulation_Time_Program.xlsx
    sheets_time = generar_simulation_time_program(
        traza_eventos, recursos_config, numero_turnos, duracion_turno, n_simulaciones)
    p = os.path.join(output_path, 'Simulation_Time_Program.xlsx')
    escribir_excel_multihoja(p, sheets_time)
    guardados.append(p)

    return guardados


def validar_marina(archivo_marina):
    """Valida y carga Input_Marina.xlsx (hoja Vaciaderos con sitios de depósito)."""
    errors, warnings_list = [], []
    try:
        xls = pd.ExcelFile(archivo_marina)
        if 'Vaciaderos' not in xls.sheet_names:
            # Intentar leer la primera hoja si no hay 'Vaciaderos'
            df = xls.parse(xls.sheet_names[0])
            warnings_list.append("Input_Marina: no se encontró hoja 'Vaciaderos', se usó la primera hoja.")
        else:
            df = xls.parse('Vaciaderos')
        req_cols = ['Nivel', 'Capacidad', 'Maximo', 'Tipo']
        missing = [c for c in req_cols if c not in df.columns]
        if missing:
            warnings_list.append(f"Input_Marina/Vaciaderos: columnas opcionales no encontradas: {', '.join(missing)}")
        return df, errors, warnings_list
    except Exception as e:
        errors.append(f"Error al leer Input_Marina: {e}")
        return pd.DataFrame(), errors, warnings_list


def validar_muckpile(archivo_muckpile):
    """Valida y carga Muckpile.xlsx (estado actual del stockpile por nivel)."""
    errors, warnings_list = [], []
    try:
        xls = pd.ExcelFile(archivo_muckpile)
        sheet = xls.sheet_names[0]
        df = xls.parse(sheet)
        req_cols = ['Nivel', 'Capacidad Ocupada', 'Maximo', 'Tipo']
        missing = [c for c in req_cols if c not in df.columns]
        if missing:
            warnings_list.append(f"Muckpile/{sheet}: columnas opcionales no encontradas: {', '.join(missing)}")
        return df, errors, warnings_list
    except Exception as e:
        errors.append(f"Error al leer Muckpile: {e}")
        return pd.DataFrame(), errors, warnings_list


# Inicializar session state
if 'datos_cargados' not in st.session_state:
    st.session_state.datos_cargados = False
if 'data_version' not in st.session_state:
    # Se incrementa cada vez que se cargan archivos nuevos y validos (Menu 1). Junto con
    # nivel_seleccionado, es la clave de cache de filtrar_actividades_nivel/recursos_nivel/
    # frentes_nivel: mientras no cambie, se reutiliza el filtrado ya calculado.
    st.session_state.data_version = 0
if 'actividades_modificadas' not in st.session_state:
    st.session_state.actividades_modificadas = {}
if 'actividades_originales' not in st.session_state:
    st.session_state.actividades_originales = pd.DataFrame() # Inicializar como DataFrame vacío
if 'actividades_adicionales' not in st.session_state:
    st.session_state.actividades_adicionales = []
if 'restricciones_geologicas' not in st.session_state:
    st.session_state.restricciones_geologicas = []
if 'mwd_config' not in st.session_state:
    st.session_state.mwd_config = None
if 'fallas_equipos' not in st.session_state:
    st.session_state.fallas_equipos = []
if 'mantenciones_programadas' not in st.session_state:
    # Mantenciones PLANIFICADAS por equipo (unidad especifica), independientes de
    # 'fallas_equipos' (que modela fallas ALEATORIAS e imprevistas por frente/recurso).
    # Ver _menu_8b_mantenciones_programadas() y su uso dentro de simular_avance_con_transporte.
    st.session_state.mantenciones_programadas = []
if 'df_marina' not in st.session_state:
    st.session_state.df_marina = pd.DataFrame()
if 'df_muckpile' not in st.session_state:
    st.session_state.df_muckpile = pd.DataFrame()


# SECCION DE INPUTS


# ----------------------------------------------------------------------
# Velocidad de viaje por recurso / por unidad (Menu 6)
# ----------------------------------------------------------------------
OPC_VEL_MISMA = "Misma velocidad para todas las unidades"
OPC_VEL_POR_UNIDAD = "Definir velocidad por unidad"


def _cfg_velocidad_defecto():
    return {'dist': 'Constante', 'params': 15.0}  # igual al valor inicial del formulario


def _preservar_widgets_vel(grupo, tag_activo):
    """Streamlit borra el estado de los widgets que no se dibujan en una corrida. Como al elegir otra
    unidad (o pasar de 'misma velocidad' a 'por unidad') los formularios de las demas dejan de dibujarse,
    se reasigna su estado para que al volver conserven lo que el usuario habia definido."""
    registro = st.session_state.get('_vel_claves', {}).get(grupo, {})
    for tag, claves in registro.items():
        if tag == tag_activo:
            continue
        for k in claves:
            if k in st.session_state:
                st.session_state[k] = st.session_state[k]


def _describir_cfg_velocidad(cfg):
    if cfg['dist'] == 'Constante':
        return f"Constante: {float(cfg['params']):.2f} km/h"
    params = cfg['params']
    txt = ", ".join(f"{k}={float(v):.2f}" for k, v in params.items() if not str(k).startswith('_'))
    if params.get('_trunc_lo') is not None:
        txt += f" · truncada a [{params['_trunc_lo']:.2f}, {params['_trunc_hi']:.2f}]"
    return f"{cfg['dist']} ({txt})"


def _form_velocidad_recurso(tag, etiqueta, grupo):
    """Dibuja el formulario de distribucion de velocidad (selector, parametros y vista previa de la PDF)
    y devuelve {'dist':..., 'params':...}. 'tag' identifica los widgets (el nombre del recurso, o
    'recurso__uNNN' para una unidad concreta); 'etiqueta' es el texto que ve el usuario."""
    claves = []

    def K(nombre):
        k = f"{nombre}_{tag}"
        if k not in claves:
            claves.append(k)
        return k

    def _V(nombre, defecto):
        # Solo pasa value= la primera vez; si el widget ya tiene estado, Streamlit lo respeta.
        return {} if f"{nombre}_{tag}" in st.session_state else {'value': defecto}

    dist_velocidad = st.selectbox(
        f"Distribución de Velocidad para {etiqueta}",
        ['Constante', 'normal', 'weibull', 'gamma', 'lognormal',
         'fisk', 'kstwobign', 'rayleigh', 'foldcauchy', 'foldnorm', 'ncx2',
         'burr', 'loglaplace', 'maxwell', 'nakagami'],
        key=K("vel_dist")
    )

    if dist_velocidad == 'Constante':
        valor = st.number_input(
            f"Valor Constante (km/h) para {etiqueta}",
            min_value=0.1, step=0.5, key=K("vel_cte"), **_V("vel_cte", 15.0)
        )
        cfg_vel = {'dist': 'Constante', 'params': valor}
    elif dist_velocidad == 'normal':
        col_vn1, col_vn2 = st.columns(2)
        with col_vn1:
            mean_val = st.number_input("mean", min_value=0.1, step=0.5, key=K("vel_mean"), **_V("vel_mean", 15.0))
        with col_vn2:
            std_val = st.number_input("std", min_value=0.01, step=0.1, key=K("vel_std"), **_V("vel_std", 3.0))
        cfg_vel = {'dist': 'normal', 'params': {'mean': mean_val, 'std': std_val}}
    elif dist_velocidad == 'weibull':
        col_vw1, col_vw2 = st.columns(2)
        with col_vw1:
            c_val = st.number_input("c", min_value=0.01, step=0.1, key=K("vel_weib_c"), **_V("vel_weib_c", 2.0))
        with col_vw2:
            scale_val = st.number_input("scale", min_value=0.01, step=0.5, key=K("vel_weib_scale"), **_V("vel_weib_scale", 15.0))
        cfg_vel = {'dist': 'weibull', 'params': {'c': c_val, 'loc': 0, 'scale': scale_val}}
    elif dist_velocidad == 'gamma':
        col_vg1, col_vg2 = st.columns(2)
        with col_vg1:
            a_val = st.number_input("a", min_value=0.01, step=0.1, key=K("vel_gamma_a"), **_V("vel_gamma_a", 2.0))
        with col_vg2:
            scale_val = st.number_input("scale", min_value=0.01, step=0.5, key=K("vel_gamma_scale"), **_V("vel_gamma_scale", 7.5))
        cfg_vel = {'dist': 'gamma', 'params': {'a': a_val, 'loc': 0, 'scale': scale_val}}
    elif dist_velocidad == 'lognormal':
        col_vl1, col_vl2 = st.columns(2)
        with col_vl1:
            s_val = st.number_input("s (sigma)", min_value=0.01, step=0.01, key=K("vel_logn_s"), **_V("vel_logn_s", 0.3))
        with col_vl2:
            scale_val = st.number_input("scale", min_value=0.01, step=0.5, key=K("vel_logn_scale"), **_V("vel_logn_scale", 14.0))
        cfg_vel = {'dist': 'lognormal', 'params': {'s': s_val, 'loc': 0, 'scale': scale_val}}
    elif dist_velocidad == 'fisk':
        col_vf1, col_vf2, col_vf3 = st.columns(3)
        with col_vf1:
            c_val = st.number_input("c", min_value=0.01, step=0.01, key=K("vel_fisk_c"), **_V("vel_fisk_c", 6.26))
        with col_vf2:
            loc_val = st.number_input("loc", step=0.1, key=K("vel_fisk_loc"), **_V("vel_fisk_loc", 0.0))
        with col_vf3:
            scale_val = st.number_input("scale", min_value=0.01, step=0.5, key=K("vel_fisk_scale"), **_V("vel_fisk_scale", 15.0))
        cfg_vel = {'dist': 'fisk', 'params': {'c': c_val, 'loc': loc_val, 'scale': scale_val}}
    elif dist_velocidad == 'kstwobign':
        col_vk1, col_vk2 = st.columns(2)
        with col_vk1:
            loc_val = st.number_input("loc", step=0.1, key=K("vel_ks_loc"), **_V("vel_ks_loc", 0.0))
        with col_vk2:
            scale_val = st.number_input("scale", min_value=0.01, step=0.5, key=K("vel_ks_scale"), **_V("vel_ks_scale", 15.0))
        cfg_vel = {'dist': 'kstwobign', 'params': {'loc': loc_val, 'scale': scale_val}}
    elif dist_velocidad == 'rayleigh':
        col_vr1, col_vr2 = st.columns(2)
        with col_vr1:
            loc_val = st.number_input("loc", step=0.1, key=K("vel_ray_loc"), **_V("vel_ray_loc", 0.0))
        with col_vr2:
            scale_val = st.number_input("scale", min_value=0.01, step=0.5, key=K("vel_ray_scale"), **_V("vel_ray_scale", 12.0))
        cfg_vel = {'dist': 'rayleigh', 'params': {'loc': loc_val, 'scale': scale_val}}
    elif dist_velocidad == 'foldcauchy':
        col_vc1, col_vc2 = st.columns(2)
        with col_vc1:
            c_val = st.number_input("c", min_value=0.0, step=0.01, key=K("vel_fc_c"), **_V("vel_fc_c", 0.0))
        with col_vc2:
            scale_val = st.number_input("scale", min_value=0.01, step=0.5, key=K("vel_fc_scale"), **_V("vel_fc_scale", 10.0))
        cfg_vel = {'dist': 'foldcauchy', 'params': {'c': c_val, 'loc': 0, 'scale': scale_val}}
    elif dist_velocidad == 'foldnorm':
        col_vfn1, col_vfn2 = st.columns(2)
        with col_vfn1:
            c_val = st.number_input("c", min_value=0.0, step=0.01, key=K("vel_fn_c"), **_V("vel_fn_c", 0.0))
        with col_vfn2:
            scale_val = st.number_input("scale", min_value=0.01, step=0.5, key=K("vel_fn_scale"), **_V("vel_fn_scale", 15.0))
        cfg_vel = {'dist': 'foldnorm', 'params': {'c': c_val, 'loc': 0, 'scale': scale_val}}
    elif dist_velocidad == 'ncx2':
        col_vx1, col_vx2, col_vx3 = st.columns(3)
        with col_vx1:
            df_val = st.number_input("df (grados libertad)", min_value=1.0, step=1.0, key=K("vel_ncx2_df"), **_V("vel_ncx2_df", 2.0))
        with col_vx2:
            nc_val = st.number_input("nc (no-centralidad)", min_value=0.0, step=0.1, key=K("vel_ncx2_nc"), **_V("vel_ncx2_nc", 0.0))
        with col_vx3:
            scale_val = st.number_input("scale", min_value=0.01, step=0.5, key=K("vel_ncx2_scale"), **_V("vel_ncx2_scale", 5.0))
        cfg_vel = {'dist': 'ncx2', 'params': {'df': df_val, 'nc': nc_val, 'loc': 0, 'scale': scale_val}}
    elif dist_velocidad == 'burr':
        col_vb1, col_vb2, col_vb3 = st.columns(3)
        with col_vb1:
            c_val = st.number_input("c", min_value=0.01, step=0.1, key=K("vel_burr_c"), **_V("vel_burr_c", 4.0))
        with col_vb2:
            d_val = st.number_input("d", min_value=0.01, step=0.1, key=K("vel_burr_d"), **_V("vel_burr_d", 2.0))
        with col_vb3:
            scale_val = st.number_input("scale", min_value=0.01, step=0.5, key=K("vel_burr_scale"), **_V("vel_burr_scale", 15.0))
        cfg_vel = {'dist': 'burr', 'params': {'c': c_val, 'd': d_val, 'loc': 0, 'scale': scale_val}}
    elif dist_velocidad == 'loglaplace':
        col_vll1, col_vll2 = st.columns(2)
        with col_vll1:
            c_val = st.number_input("c", min_value=0.01, step=0.1, key=K("vel_logl_c"), **_V("vel_logl_c", 2.0))
        with col_vll2:
            scale_val = st.number_input("scale", min_value=0.01, step=0.5, key=K("vel_logl_scale"), **_V("vel_logl_scale", 15.0))
        cfg_vel = {'dist': 'loglaplace', 'params': {'c': c_val, 'loc': 0, 'scale': scale_val}}
    elif dist_velocidad == 'maxwell':
        scale_val = st.number_input("scale", min_value=0.01, step=0.5, key=K("vel_maxwell_scale"), **_V("vel_maxwell_scale", 10.0))
        cfg_vel = {'dist': 'maxwell', 'params': {'loc': 0, 'scale': scale_val}}
    elif dist_velocidad == 'nakagami':
        col_vnk1, col_vnk2 = st.columns(2)
        with col_vnk1:
            nu_val = st.number_input("nu", min_value=0.01, step=0.1, key=K("vel_naka_nu"), **_V("vel_naka_nu", 1.0))
        with col_vnk2:
            scale_val = st.number_input("scale", min_value=0.01, step=0.5, key=K("vel_naka_scale"), **_V("vel_naka_scale", 15.0))
        cfg_vel = {'dist': 'nakagami', 'params': {'nu': nu_val, 'loc': 0, 'scale': scale_val}}

    # ---- Vista previa de la PDF + filtro de área (igual que en el Menú 8) ----
    # Ayuda a validar visualmente la forma de la distribución elegida y a
    # explorar "qué pasa si solo miro las velocidades mayores/menores/entre
    # tales valores". Si se deja un filtro activo, ese rango se guarda como
    # truncamiento real ('_trunc_lo'/'_trunc_hi'): generar_velocidad_batch()
    # ya sabe leer estas claves e invertir la CDF restringida a ese tramo, así
    # que la simulación usa velocidades garantizadas dentro del rango elegido.
    if dist_velocidad == 'Constante':
        st.caption(f"Velocidad constante: siempre **{valor:.2f} km/h**. No aplica curva de densidad.")
    else:
        st.markdown("##### Vista Previa de la Distribución (PDF):")
        rango_elegido = vista_previa_pdf_con_filtro(
            dist_velocidad,
            cfg_vel['params'],
            key_sufijo=f"recurso_{tag}",
            unidad="km/h",
            titulo_eje_x="Velocidad (km/h)"
        )
        if rango_elegido is not None:
            cfg_vel['params']['_trunc_lo'] = rango_elegido[0]
            cfg_vel['params']['_trunc_hi'] = rango_elegido[1]

    for n in ("modo_area", "corte_mayor", "corte_menor", "corte_rango"):
        k = f"{n}_recurso_{tag}"
        if k not in claves:
            claves.append(k)
    st.session_state.setdefault('_vel_claves', {}).setdefault(grupo, {})[tag] = claves
    return cfg_vel


# ======================================================================
# #####INTERFAZ STREAMLIT — MENÚS 1 A 16#####
# ======================================================================
st.header("Parámetros de Entrada")

# ----------------------------------------------------
# Cargar Datos 
# ----------------------------------------------------

# ======================================================================
# #####Menu 1 — Cargar Datos#####
# ======================================================================
with seccion_menu('1', 'Cargar Datos', ayuda=(
        "Archivos requeridos: 1) Operación (obligatorio): Input_Actividades_Recursos_Frentes.xlsx, con hojas "
        "Actividades, Recursos y Frentes. 2) Restricciones (obligatorio): Input_Restricciones.xlsx, con hojas "
        "FrenteHundimiento, ObrasCiviles y Restriccion. 3) Vaciaderos (opcional): Input_Marina.xlsx, hoja Vaciaderos. "
        "4) Muckpile (opcional): Muckpile.xlsx, estado del stockpile por nivel.")):

    col_file1, col_file2 = st.columns(2)

    with col_file1:
        st.markdown("**Archivo 1 (Operación):** `Input_Actividades_Recursos_Frentes.xlsx`")
        archivo_main = st.file_uploader("Cargar archivo de Operación", type=['xlsx', 'xls'], key="file_main")

    with col_file2:
        st.markdown("**Archivo 2 (Restricciones):** `Input_Restricciones.xlsx`")
        archivo_restricciones = st.file_uploader("Cargar archivo de Restricciones", type=['xlsx', 'xls'], key="file_restricciones")

    col_file3, col_file4 = st.columns(2)
    with col_file3:
        st.markdown("**Archivo 3 – Vaciaderos** *(opcional)*: `Input_Marina.xlsx`")
        archivo_marina = st.file_uploader("Cargar Input_Marina (Vaciaderos)", type=['xlsx', 'xls'], key="file_marina")

    with col_file4:
        st.markdown("**Archivo 4 – Muckpile** *(opcional)*: `Muckpile.xlsx`")
        archivo_muckpile_file = st.file_uploader("Cargar Muckpile (Stockpile actual)", type=['xlsx', 'xls'], key="file_muckpile")

    # Cargar archivos opcionales
    if archivo_marina:
        df_marina_loaded, errs_marina, warns_marina = validar_marina(archivo_marina)
        if errs_marina:
            for e in errs_marina:
                st.error(e)
        else:
            st.session_state.df_marina = df_marina_loaded
            for w in warns_marina:
                st.warning(w)
            st.success(f"Input_Marina cargado: {len(df_marina_loaded)} vaciaderos encontrados.")

    if archivo_muckpile_file:
        df_muckpile_loaded, errs_mp, warns_mp = validar_muckpile(archivo_muckpile_file)
        if errs_mp:
            for e in errs_mp:
                st.error(e)
        else:
            st.session_state.df_muckpile = df_muckpile_loaded
            for w in warns_mp:
                st.warning(w)
            st.success(f"Muckpile cargado: {len(df_muckpile_loaded)} registros de stockpile encontrados.")

    if archivo_main and archivo_restricciones:
        try:
            datos_validados, errores_input, warnings_input = cargar_y_validar_inputs(archivo_main, archivo_restricciones)

            if errores_input:
                st.session_state.datos_cargados = False
                for error in errores_input:
                    st.error(error)
            else:
                st.session_state.df_actividades = datos_validados['df_actividades']
                st.session_state.df_recursos = datos_validados['df_recursos']
                st.session_state.df_frentes = datos_validados['df_frentes']
                st.session_state.df_fh = datos_validados['df_fh']
                st.session_state.df_oc = datos_validados['df_oc']
                st.session_state.df_res = datos_validados['df_res']
                st.session_state.actividades_originales = st.session_state.df_actividades.copy()
                st.session_state.datos_cargados = True
                st.session_state.data_version += 1
                st.success("Datos cargados y validados correctamente.")
                log_evento('INFO', f"Datos validados: {len(st.session_state.df_actividades)} actividades, "
                                   f"{len(st.session_state.df_recursos)} recursos, {len(st.session_state.df_frentes)} frentes", dedupe=True)
                for warning in warnings_input:
                    st.warning(warning)

        except Exception as e:
            st.session_state.datos_cargados = False
            st.error(f"Error al cargar datos. Revise hojas, columnas y formatos. Detalle: {e}")

# ----------------------------------------------------
# Vista de Transparencia - Bases de Datos Cargadas (los 4 archivos, hoja por hoja)
# ----------------------------------------------------
_archivos_subidos = [
    ("Archivo 1 – Operación", archivo_main),
    ("Archivo 2 – Restricciones", archivo_restricciones),
    ("Archivo 3 – Vaciaderos (Input_Marina)", archivo_marina),
    ("Archivo 4 – Muckpile (Stockpile actual)", archivo_muckpile_file),
]
_archivos_disponibles = [(nombre, arch) for nombre, arch in _archivos_subidos if arch is not None]

if _archivos_disponibles:

    # ======================================================================
    # #####Menu 2 — Vista de Transparencia (Bases de Datos Cargadas)#####
    # ======================================================================
    with seccion_menu('2', 'Explorador de Datos', ayuda="Muestra cada hoja de los Excel cargados, tal como se subieron. Use las pestañas para cambiar de archivo."):

        tabs_archivos = st.tabs([nombre for nombre, _ in _archivos_disponibles])

        for tab, (nombre, archivo) in zip(tabs_archivos, _archivos_disponibles):
            with tab:
                try:
                    archivo.seek(0)
                    excel_file_obj = pd.ExcelFile(archivo)
                    nombres_hojas = excel_file_obj.sheet_names

                    if not nombres_hojas:
                        st.info("Este archivo no contiene hojas legibles.")
                    else:
                        st.caption(f"`{archivo.name}` — {len(nombres_hojas)} hoja(s): {', '.join(nombres_hojas)}")
                        sub_tabs_hojas = st.tabs(nombres_hojas)
                        for sub_tab, hoja in zip(sub_tabs_hojas, nombres_hojas):
                            with sub_tab:
                                try:
                                    df_hoja = excel_file_obj.parse(hoja)
                                    mostrar_tabla(df_hoja, width='stretch', hide_index=True)
                                    st.caption(f"{df_hoja.shape[0]} filas × {df_hoja.shape[1]} columnas.")
                                except Exception as e_hoja:
                                    st.warning(f"No se pudo leer la hoja '{hoja}': {e_hoja}")
                except Exception as e_archivo:
                    st.error(f"No se pudo leer el archivo '{archivo.name}': {e_archivo}")
                finally:
                    try:
                        archivo.seek(0)
                    except Exception:
                        pass

# ----------------------------------------------------
# Vista de Vaciaderos y Muckpile (si se cargaron)
# ----------------------------------------------------
if not st.session_state.df_marina.empty or not st.session_state.df_muckpile.empty:

    # ======================================================================
    # #####Menu 3 — Vaciaderos y Estado del Stockpile (Muckpile)#####
    # ======================================================================
    with seccion_menu('3', 'Vaciaderos y Estado del Stockpile (Muckpile)'):
        col_mar, col_mp = st.columns(2)

        with col_mar:
            if not st.session_state.df_marina.empty:
                mostrar_titulo_centrado("Vaciaderos (Input_Marina)", "Puntos de descarga del material.")
                if 'Capacidad' in st.session_state.df_marina.columns and 'Maximo' in st.session_state.df_marina.columns:
                    total_cap = st.session_state.df_marina['Capacidad'].sum()
                    total_max = st.session_state.df_marina['Maximo'].sum()
                    mostrar_tarjetas_metricas(
                        [("Capacidad total", f"{total_cap:.0f}", "", True, f"Máximo: {total_max:.0f}")], ancho_max=260
                    )
            else:
                st.info("No se cargó Input_Marina.")

        with col_mp:
            if not st.session_state.df_muckpile.empty:
                mostrar_titulo_centrado("Stockpile Actual (Muckpile)", "Estado actual del material almacenado por nivel/ubicación.")
                if 'Capacidad Ocupada' in st.session_state.df_muckpile.columns and 'Maximo' in st.session_state.df_muckpile.columns:
                    ocupado = st.session_state.df_muckpile['Capacidad Ocupada'].sum()
                    maximo = st.session_state.df_muckpile['Maximo'].sum()
                    pct = (ocupado / maximo * 100) if maximo > 0 else 0
                    mostrar_tarjetas_metricas(
                        [("Ocupación total", f"{ocupado:.0f} / {maximo:.0f}", "", True, f"{pct:.1f}% lleno")], ancho_max=260
                    )
                    if pct > 80:
                        st.warning("Stockpile sobre 80% de capacidad. Puede restringir operaciones de extracción.")
            else:
                st.info("No se cargó Muckpile.")

# ----------------------------------------------------
# Seleccionar Nivel
# ----------------------------------------------------
if st.session_state.datos_cargados:

    # ======================================================================
    # #####Menu 4 — Seleccionar Nivel#####
    # ======================================================================
    with seccion_menu('4', 'Seleccionar Nivel'):
        niveles_disponibles = st.session_state.df_actividades['Nivel'].unique()
        nivel_seleccionado = st.selectbox("Nivel", niveles_disponibles)

    if nivel_seleccionado:
        actividades_nivel = filtrar_actividades_nivel(
            st.session_state.df_actividades, nivel_seleccionado, st.session_state.data_version
        )
        recursos_nivel = filtrar_recursos_nivel(
            st.session_state.df_recursos, nivel_seleccionado, st.session_state.data_version
        )
        frentes_nivel = filtrar_frentes_nivel(
            st.session_state.df_frentes, nivel_seleccionado, st.session_state.data_version
        )

        # Aplicar modificaciones guardadas
        for idx, modificacion in st.session_state.actividades_modificadas.items():
            if idx in actividades_nivel.index:
                actividades_nivel.at[idx, 'Distribucion'] = modificacion['Distribucion']
                actividades_nivel.at[idx, 'Tiempo'] = modificacion['Tiempo']

        # --- INSERCIÓN: Agregar Actividades Personalizadas ---
        if 'Es_Personalizada' not in actividades_nivel.columns:
            actividades_nivel['Es_Personalizada'] = False
        if 'Frentes_Aplicables' not in actividades_nivel.columns:
            actividades_nivel['Frentes_Aplicables'] = None # Se asume aplicable a todos si es None/False
        
        # Copia del listado base (con su secuencia original) para el Menú de actividades
        # personalizadas: ahí se usa para mostrar cómo quedaría el orden al insertar.
        st.session_state['_act_base_nivel'] = actividades_nivel.loc[
            actividades_nivel['Actividad'].notna(), ['Secuencia', 'Actividad', 'Recurso']].copy()

        if st.session_state.actividades_adicionales:
            # Cada personalizada define su posición (o va al final); las existentes se corren.
            actividades_nivel, _nuevas_con_secuencia = aplicar_secuencias_personalizadas(
                actividades_nivel, st.session_state.actividades_adicionales)
            df_nuevas_act = pd.DataFrame(_nuevas_con_secuencia)
            
            # Asegurar que las columnas existan en las nuevas actividades para el concat
            for col in actividades_nivel.columns:
                if col not in df_nuevas_act.columns:
                    df_nuevas_act[col] = np.nan
            
            # Unir actividades: cada nueva trae la Secuencia que el usuario eligió
            actividades_nivel = pd.concat([actividades_nivel, df_nuevas_act], ignore_index=True)

        # --- FIN INSERCIÓN ---

        # ----------------------------------------------------
        # Parametros Generales
        # ----------------------------------------------------
        # Toda la sección vive dentro de un @st.fragment: así, cambiar el sistema de
        # turnos, sincronizar turnos/horas, o presionar "Aplicar parámetros" solo
        # vuelve a ejecutar ESTE bloque (rápido) en vez de re-correr todo el script
        # (~4600 líneas, con llamadas a Redis, pandas y Plotly de las secciones de
        # abajo) en cada interacción. Eso es lo que antes hacía "recargar toda la
        # app" y se veía como si secciones como la 7 se vaciaran/reiniciaran.
        # Los valores finales se dejan en st.session_state para que el resto del
        # script (fuera del fragment) los siga leyendo con normalidad.
        @st.fragment
        def _render_parametros_generales():

            # ======================================================================
            # #####Menu 5 — Parámetros Generales#####
            # ======================================================================
            with seccion_menu('5', 'Parámetros Generales'):

                def _duracion_turno_actual():
                    return 12.0 if '12x2' in st.session_state.get('sistema_turnos', '12x2 (2 turnos de 12h)') else 8.0

                def _sync_horas_desde_turnos():
                    """Al cambiar 'Numero de Turnos del Programa', recalcula 'Tiempo Limite (horas)' = turnos x duracion del turno."""
                    dur = _duracion_turno_actual()
                    num = st.session_state.get('numero_turnos_input', 1)
                    st.session_state['tiempo_limite_input'] = float(num * dur)

                def _sync_turnos_desde_horas():
                    """Al cambiar 'Tiempo Limite (horas)', recalcula 'Numero de Turnos del Programa' = horas / duracion del turno."""
                    dur = _duracion_turno_actual()
                    tiempo = st.session_state.get('tiempo_limite_input', dur)
                    st.session_state['numero_turnos_input'] = max(1, round(tiempo / dur))

                sistema_turnos = st.radio(
                    "Sistema de Turnos", 
                    ['12x2 (2 turnos de 12h)', '8x3 (3 turnos de 8h)'], 
                    key="sistema_turnos", 
                    horizontal=True,
                    on_change=_sync_horas_desde_turnos  # al cambiar el sistema, se recalculan las horas manteniendo el N° de turnos
                )
                if '12x2' in sistema_turnos:
                    turnos_info = "2 turnos de 12 horas."
                    duracion_turno_base = 12.0
                else:
                    turnos_info = "3 turnos de 8 horas."
                    duracion_turno_base = 8.0
                st.info(turnos_info)

                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    numero_turnos = st.number_input(
                        "Numero de Turnos del Programa", min_value=1, value=62, step=1,
                        key="numero_turnos_input", on_change=_sync_horas_desde_turnos,
                        help="Al modificarlo, el 'Tiempo Limite (horas)' se recalcula automáticamente (N° turnos × duración del turno)."
                    )
                with col_p2:
                    tiempo_limite = st.number_input(
                        "Tiempo Limite (horas)",
                        min_value=1.0,
                        value=float(numero_turnos * duracion_turno_base),
                        step=1.0,
                        key="tiempo_limite_input",
                        on_change=_sync_turnos_desde_horas,
                        help="En el motor original equivale a numero de turnos x horas por turno. Al modificarlo, el "
                             "'Numero de Turnos del Programa' se recalcula automáticamente (horas ÷ duración del turno), "
                             "y viceversa: ambos campos quedan sincronizados."
                    )

                # Antes estos 3 parámetros vivían dentro de un st.form y solo se
                # aplicaban al presionar "Aplicar parámetros". Se saca el form: al vivir
                # sueltos DENTRO del fragment, cada cambio dispara un rerun (rápido) de
                # solo este bloque y el valor queda aplicado de inmediato, sin necesidad
                # de ningún botón ni de pasar por este menú explícitamente.
                col_p3, col_p4 = st.columns(2)
                with col_p3:
                    metros_avance = st.number_input("Metros por Ciclo por Defecto", min_value=0.1, value=3.5, step=0.1, key="metros_avance_input")
                with col_p4:
                    n_simulaciones = st.number_input("Número de Simulaciones Monte Carlo", min_value=1, value=1000, step=100, key="n_simulaciones_input")

                _opciones_ruta_critica = sorted(frentes_nivel['Frentes'].dropna().unique())
                # Si el usuario cambió de Nivel (Menú 4), el valor previamente guardado
                # bajo esta key puede traer frentes que ya no existen en las opciones
                # actuales; se reconcilia ANTES de crear el widget para evitar el error
                # de Streamlit "default value not part of options".
                if 'ruta_critica_input' in st.session_state:
                    st.session_state['ruta_critica_input'] = [
                        f for f in st.session_state['ruta_critica_input'] if f in _opciones_ruta_critica
                    ]
                ruta_critica = st.multiselect(
                    "Ruta critica / frentes prioritarios",
                    _opciones_ruta_critica,
                    # Se filtra contra las opciones vigentes para no romper si el usuario
                    # cambió de Nivel (Menú 4) y algún frente de la selección anterior ya
                    # no existe en este Nivel.
                    default=[f for f in st.session_state.get('ruta_critica', []) if f in _opciones_ruta_critica],
                    key="ruta_critica_input",
                    help="Opcional. Elija los frentes en el orden deseado: el primero que seleccione tiene la máxima prioridad (1º), luego el 2º, 3º, etc. Los frentes no incluidos se simulan después, ordenados por cercanía a FH/OOCC. Para cambiar el orden, quite un frente y vuelva a agregarlo."
                )

                seccion_col = columna_seccion(frentes_nivel)
                metros_por_seccion = {}
                if seccion_col:
                    # Opcional y apagada por defecto: si el usuario no la activa, todos los
                    # frentes usan el unico 'Metros por Ciclo por Defecto' de arriba, exactamente
                    # igual que antes de esta funcionalidad. Fuera del form (no dentro del
                    # st.form de arriba) para que el data_editor aparezca/desaparezca al tiro,
                    # sin tener que apretar 'Aplicar parametros'.
                    usar_metros_variables = st.checkbox(
                        "Usar Metros por Ciclo variable por Sección",
                        value=st.session_state.get('usar_metros_variables_seccion', False),
                        key="usar_metros_variables_seccion_input",
                        help="Cada frente trae su sección/dimensión (ej. '4.2x4.1') desde la hoja Frentes "
                             "del Archivo 1. Con esta opción activada puede definir un avance por ciclo "
                             "distinto para cada sección; los valores no vienen precargados con ningún "
                             "numero calibrado, usted los define. Desactivada, todas las secciones avanzan "
                             "el 'Metros por Ciclo por Defecto' de arriba."
                    )
                    st.session_state.usar_metros_variables_seccion = usar_metros_variables

                    if usar_metros_variables:
                        secciones = sorted(frentes_nivel[seccion_col].dropna().astype(str).unique())
                        st.markdown("#### Metros por Ciclo por Sección (dimensión de excavación)")
                        # Unico default al mostrar la tabla por primera vez (o para una seccion
                        # nueva que no se ha editado aun): el mismo generico de arriba, nunca un
                        # valor calibrado hardcodeado. Si el usuario ya edito una seccion en un
                        # rerun anterior, se respeta lo que el mismo ingreso.
                        metros_seccion_previos = st.session_state.get('metros_por_seccion', {})
                        df_metros_seccion = pd.DataFrame({
                            'Sección': secciones,
                            'Metros por Ciclo': [
                                float(metros_seccion_previos.get(seccion, metros_avance))
                                for seccion in secciones
                            ]
                        })
                        df_metros_seccion_editado = st.data_editor(
                            df_metros_seccion,
                            hide_index=True,
                            width='stretch',
                            key="metros_seccion_editor",
                            disabled=['Sección'],
                            column_config={
                                'Sección': st.column_config.TextColumn('Sección', alignment="center"),
                                'Metros por Ciclo': st.column_config.NumberColumn(
                                    min_value=0.01, step=0.1, format="%.2f", alignment="center"
                                )
                            }
                        )
                        metros_por_seccion = {
                            str(row['Sección']): float(row['Metros por Ciclo'])
                            for _, row in df_metros_seccion_editado.iterrows()
                            if pd.notna(row['Sección']) and pd.notna(row['Metros por Ciclo'])
                        }

                # ---- Demoras programadas por cambio de turno (opcional) ----
                st.markdown("#### Demoras programadas (cambio de turno)")
                usar_demora_turno = st.checkbox(
                    "Aplicar una demora programada en cada cambio de turno",
                    value=st.session_state.get('usar_demora_cambio_turno', False),
                    key="usar_demora_cambio_turno_input",
                    help="Opcional. Al terminar cada turno (cada 12 h en 12x2, cada 8 h en 8x3) todos los equipos se "
                         "detienen una demora sorteada de la distribución que elija abajo; luego parte el turno "
                         "siguiente y vuelve a ocurrir al final de ese. El trabajo en curso se pausa y se reanuda "
                         "al terminar la demora, así que alarga ciclos y avances y aparece en todos los resultados."
                )
                st.session_state.usar_demora_cambio_turno = usar_demora_turno
                demoras_cambio_turno = None
                if usar_demora_turno:
                    dist_turno = st.selectbox(
                        "Distribución de la demora por cambio de turno (horas)",
                        list(DISTRIBUTION_PARAM_INFO.keys()),
                        index=0, key="demora_turno_dist_input"
                    )
                    info_dist_turno = DISTRIBUTION_PARAM_INFO[dist_turno]
                    valores_dist_turno = {}
                    cols_pt = st.columns(min(3, len(info_dist_turno['parametros'])) or 1)
                    for i_p, p_ in enumerate(info_dist_turno['parametros']):
                        with cols_pt[i_p % len(cols_pt)]:
                            valores_dist_turno[p_['key']] = st.number_input(
                                p_['label'], min_value=p_.get('min'),
                                value=float(p_['default']) if dist_turno != 'Cte' else 0.5,
                                step=0.01 if p_.get('min') == 0.0001 else 0.1,
                                key=f"demora_turno_param_{dist_turno}_{p_['key']}"
                            )
                    params_turno = valores_dist_turno['valor'] if dist_turno == 'Cte' else valores_dist_turno
                    if dist_turno == 'Cte':
                        st.info(f"Distribución constante: cada cambio de turno demora siempre **{params_turno:.3f} h**.")
                    else:
                        rango_turno = vista_previa_pdf_con_filtro(
                            dist_turno, params_turno, key_sufijo="demora_cambio_turno", unidad="h",
                            titulo_eje_x="Demora por cambio de turno (h)"
                        )
                        if rango_turno is not None and isinstance(params_turno, dict):
                            params_turno['_trunc_lo'] = float(rango_turno[0])
                            params_turno['_trunc_hi'] = float(rango_turno[1])
                    demoras_cambio_turno = {'activo': True, 'distribucion': dist_turno, 'params': params_turno}
                    n_cambios_turno = max(0, math.ceil(float(tiempo_limite) / float(duracion_turno_base)) - 1)
                    st.caption(f"Se aplicará al final de cada turno de {duracion_turno_base:g} h "
                               f"({n_cambios_turno} cambios de turno dentro del horizonte de {float(tiempo_limite):g} h).")
                st.session_state.demoras_cambio_turno = demoras_cambio_turno

                st.session_state.tiempo_limite = tiempo_limite
                st.session_state.metros_avance = metros_avance
                st.session_state.n_simulaciones = n_simulaciones
                st.session_state.numero_turnos = numero_turnos
                st.session_state.ruta_critica = ruta_critica
                st.session_state.metros_por_seccion = metros_por_seccion
                st.session_state.duracion_turno_base = duracion_turno_base

                # ---- Aplicación automática + feedback (reemplaza a "Aplicar parámetros") ----
                # Se compara la configuración actual contra la última "firma" registrada.
                # - Si cambió algo Y ya había una simulación corrida con la configuración
                #   anterior (ej. se simuló con 200 y se acaba de bajar a 100), esa
                #   simulación queda obsoleta: se reinicia sola y se avisa con un warning.
                # - Si cambió algo y todavía no se había simulado, solo se confirma con un
                #   success que el cambio ya quedó aplicado (nadie tiene que apretar nada).
                firma_actual = (
                    int(numero_turnos), round(float(tiempo_limite), 4), round(float(metros_avance), 4),
                    int(n_simulaciones), tuple(ruta_critica),
                    tuple(sorted((k, round(v, 4)) for k, v in metros_por_seccion.items())),
                    sistema_turnos,
                    repr(demoras_cambio_turno),
                )
                firma_previa = st.session_state.get('_firma_parametros_generales')
                if firma_previa is not None and firma_actual != firma_previa:
                    cambios = []
                    if firma_previa[0] != firma_actual[0]:
                        cambios.append(f"N° de Turnos: {firma_previa[0]} → {firma_actual[0]}")
                    if firma_previa[1] != firma_actual[1]:
                        cambios.append(f"Tiempo Límite: {firma_previa[1]:.1f} h → {firma_actual[1]:.1f} h")
                    if firma_previa[2] != firma_actual[2]:
                        cambios.append(f"Metros por Ciclo: {firma_previa[2]:.2f} → {firma_actual[2]:.2f}")
                    if firma_previa[3] != firma_actual[3]:
                        cambios.append(f"N° Simulaciones Monte Carlo: {firma_previa[3]} → {firma_actual[3]}")
                    if firma_previa[4] != firma_actual[4]:
                        cambios.append("ruta crítica/frentes prioritarios")
                    if firma_previa[5] != firma_actual[5]:
                        cambios.append("metros por ciclo por sección")
                    if firma_previa[6] != firma_actual[6]:
                        cambios.append(f"sistema de turnos: {firma_previa[6]} → {firma_actual[6]}")
                    if len(firma_previa) > 7 and firma_previa[7] != firma_actual[7]:
                        cambios.append("demora programada por cambio de turno")
                    detalle = "; ".join(cambios) if cambios else "los Parámetros Generales"

                    if 'resultados' in st.session_state:
                        reiniciar_resultados_simulacion(f"cambiaste {detalle} (Parámetros Generales)")
                    else:
                        st.success(f"Parámetros aplicados automáticamente: {detalle}.")
                st.session_state['_firma_parametros_generales'] = firma_actual

        _render_parametros_generales()

        # ----------------------------------------------------
        # Configuración de Recursos
        # ----------------------------------------------------

        # ======================================================================
        # #####Menu 6 — Configuración de Recursos#####
        # ======================================================================
        with seccion_menu('6', 'Configuración de Recursos'):
            recursos_config = {}
            velocidades_config = {}

            for _, row in recursos_nivel.iterrows():
                recurso = row['Recurso']
                col_r1, col_r2 = st.columns([1, 2])
                cantidad_base = float(row['Cantidad']) if pd.notna(row['Cantidad']) else 0.0
                cantidad_default = math.ceil(cantidad_base) if cantidad_base > 0 else 0
            
                with col_r1:
                    st.markdown(f"**{recurso}**")
                    cantidad = st.number_input(
                        f"Cantidad de {recurso}", 
                        min_value=0,
                        value=int(cantidad_default),
                        step=1,
                        key=f"recurso_qty_{recurso}",
                        help="Si el archivo trae cantidad fraccional, se redondea hacia arriba para esta simulacion discreta de equipos."
                    )
                    recursos_config[recurso] = {'cantidad': cantidad}
            
                with col_r2:
                    st.markdown("**Velocidad de Viaje (km/h)**")
                    n_unidades = int(cantidad)
                    modo_por_unidad = False
                    if n_unidades >= 2:
                        modo_vel = st.radio(
                            f"Velocidad de viaje de las {n_unidades} unidades de {recurso}",
                            [OPC_VEL_MISMA, OPC_VEL_POR_UNIDAD],
                            key=f"vel_modo_{recurso}", horizontal=True,
                            help="'Misma velocidad': todas las unidades usan una sola distribución. "
                                 "'Por unidad': cada unidad (Unidad #1, #2, ...) tiene su propia distribución de velocidad."
                        )
                        modo_por_unidad = (modo_vel == OPC_VEL_POR_UNIDAD)

                    if not modo_por_unidad:
                        _preservar_widgets_vel(recurso, recurso)
                        velocidades_config[recurso] = _form_velocidad_recurso(recurso, recurso, recurso)
                    else:
                        almacen_unidades = st.session_state.setdefault('_vel_cfg_unidades', {}).setdefault(recurso, {})
                        unidad_sel = st.selectbox(
                            "Unidad a configurar", list(range(1, n_unidades + 1)),
                            format_func=lambda i, r=recurso: f"{r} #{i}",
                            key=f"vel_unidad_sel_{recurso}"
                        )
                        tag_unidad = f"{recurso}__u{unidad_sel:03d}"
                        _preservar_widgets_vel(recurso, tag_unidad)
                        almacen_unidades[unidad_sel - 1] = _form_velocidad_recurso(tag_unidad, f"{recurso} #{unidad_sel}", recurso)

                        por_unidad = {k: almacen_unidades.get(k, _cfg_velocidad_defecto()) for k in range(n_unidades)}
                        velocidades_config[recurso] = {
                            'dist': por_unidad[0]['dist'], 'params': por_unidad[0]['params'],
                            'por_unidad': por_unidad
                        }
                        st.caption(
                            "Resumen de velocidades por unidad. Las unidades que aún no abriste usan la distribución por defecto. "
                            "Las unidades que se incorporen después por el plan de flota (Menú 7) usan la velocidad de la Unidad #1."
                        )
                        mostrar_tabla(pd.DataFrame([
                            {'Unidad': f"{recurso} #{k + 1}",
                             'Velocidad (km/h)': _describir_cfg_velocidad(por_unidad[k]),
                             'Estado': 'Configurada' if k in almacen_unidades else 'Por defecto'}
                            for k in range(n_unidades)
                        ]), width='stretch', hide_index=True)

                st.markdown("---")
        
            # Precarga de cambios de recursos por turno definidos en el archivo (columnas
            # Cambio / Turno_Cambio / Cantidad_Cambio), como punto de partida editable del
            # plan de orquestacion de flota por turno.
            #
            # Turno_Cambio y Cantidad_Cambio admiten dos formatos de celda:
            #   - Un numero simple (ej. Turno_Cambio="50", Cantidad_Cambio="4"): un solo
            #     cambio, PERMANENTE, de cantidad desde ese turno en adelante.
            #   - Una cadena de cambios secuenciales separados por "-" (ej. Turno_Cambio =
            #     "7-12", Cantidad_Cambio = "1-3"): cada turno de la lista fija la cantidad
            #     correspondiente de la lista de cantidades, y esa cantidad queda vigente
            #     hasta el siguiente cambio (o indefinidamente si es el ultimo). Ej. con
            #     Cantidad base=2, Turno_Cambio="7-12", Cantidad_Cambio="1-3": el recurso
            #     tiene 2 unidades hasta el turno 6, pasa a 1 unidad en el turno 7 (hasta el
            #     turno 11), y pasa a 3 unidades desde el turno 12 en adelante, permanente,
            #     hasta que se defina otro cambio. NO revierte solo a la cantidad original.
            #   Ambas columnas deben traer la misma cantidad de valores (mismo N de "-").
            def _parse_lista_cambio(valor):
                """Convierte un numero simple o una cadena 'a-b-c-...' en una lista de floats,
            en el orden en que vienen escritos. Acepta '-', '–', '—' o ' a ' como separador."""
                if pd.isna(valor):
                    return []
                texto = str(valor).strip()
                for sep in ['–', '—', ' a ', ' A ']:
                    texto = texto.replace(sep, '-')
                partes = [p for p in texto.split('-') if p.strip() != '']
                try:
                    return [float(p) for p in partes]
                except (ValueError, TypeError):
                    return []

            if 'plan_recursos_turno' not in st.session_state:
                st.session_state.plan_recursos_turno = []
                for _, row in recursos_nivel.iterrows():
                    if pd.notna(row.get('Cambio')) and str(row.get('Cambio')).strip().upper() in ('TRUE', '1', 'SI', 'SÍ', 'YES'):
                        turnos_cambio = _parse_lista_cambio(row.get('Turno_Cambio'))
                        cantidades_cambio = _parse_lista_cambio(row.get('Cantidad_Cambio'))
                        if not turnos_cambio or not cantidades_cambio:
                            continue
                        recurso_nombre = row['Recurso']
                        # Si alguna de las 2 columnas trae menos valores que la otra, se
                        # emparejan solo hasta el largo mas corto (el resto se ignora).
                        for turno_k, cantidad_k in zip(turnos_cambio, cantidades_cambio):
                            st.session_state.plan_recursos_turno.append({
                                'recurso': recurso_nombre,
                                'turno': int(turno_k),
                                'cantidad_nueva': int(cantidad_k),
                                'demora_horas': 0.0
                            })

            st.session_state.recursos_config = recursos_config
            st.session_state.velocidades_config = velocidades_config

            # ---- Aplicación automática + feedback (Menu 6: cantidades y velocidades) ----
            # Igual que en el Menu 5: estos widgets ya viven fuera de cualquier form (aplican
            # solos), pero si cambian DESPUES de haber corrido una simulación, esta se
            # reinicia sola para no dejar en pantalla resultados calculados con la
            # configuración de recursos anterior.
            def _firma_recursos(recursos_cfg, velocidades_cfg):
                firma_cant = tuple(sorted((r, v.get('cantidad')) for r, v in recursos_cfg.items()))
                def _firma_cfg(c):
                    p = c.get('params')
                    return (c.get('dist'), tuple(sorted(p.items())) if isinstance(p, dict) else p)
                firma_vel = tuple(sorted(
                    (r, _firma_cfg(v), tuple(sorted((k, _firma_cfg(c)) for k, c in (v.get('por_unidad') or {}).items())))
                    for r, v in velocidades_cfg.items()
                ))
                return (firma_cant, firma_vel)

            firma_recursos_actual = _firma_recursos(recursos_config, velocidades_config)
            firma_recursos_previa = st.session_state.get('_firma_recursos_config')
            if firma_recursos_previa is not None and firma_recursos_actual != firma_recursos_previa:
                if 'resultados' in st.session_state:
                    reiniciar_resultados_simulacion("cambiaste la configuración de Recursos (Configuración de Recursos: cantidades o velocidades)")
                else:
                    st.success("Configuración de Recursos (cantidades/velocidades) aplicada automáticamente.")
            st.session_state['_firma_recursos_config'] = firma_recursos_actual

        # ----------------------------------------------------
        # Orquestacion de Flota por Turno
        # ----------------------------------------------------
        # Todo el Menu 7 vive dentro de un @st.fragment, igual que los Menus 5 y 16:
        # el form ya evitaba rerun tecla por tecla, pero al presionar "Añadir cambio al
        # plan" o los botones de eliminar/limpiar, disparaba un rerun de la app COMPLETA
        # (Menus 1 a 22). Encerrado en el fragment, esos botones solo vuelven a ejecutar
        # este bloque.
        @st.fragment
        def _menu_4b_orquestacion_flota(recursos_nivel):

            # ======================================================================
            # #####Menu 7 — Orquestación de Flota por Turno u Hora#####
            # ======================================================================
            with seccion_menu('7', 'Orquestación de Flota por Turno u Hora', ayuda=(
                    "Defina, recurso por recurso, cómo cambia su cantidad disponible durante el programa. Cada cambio se "
                    "puede ubicar por TURNO (ej: bajar de 3 a 2 cuadrillas en el turno 50) o por HORA exacta de la "
                    "simulación (ej: subir a 4 jumbos en la hora 130). Si un recurso queda en 0 unidades, los frentes "
                    "que dependen de él no avanzan.")):

                # Duración de turno y horizonte del programa: permiten convertir entre turnos y horas.
                dur_turno_plan = float(st.session_state.get('duracion_turno_base', 8.0))
                n_turnos_plan = int(st.session_state.get('numero_turnos', 1))
                horizonte_plan_h = n_turnos_plan * dur_turno_plan

                def _cantidad_vigente_antes_de(recurso, inicio_h, cantidad_base):
                    """Cantidad que tendría el recurso justo antes de la hora 'inicio_h', según los cambios ya
                configurados en el plan (por turno o por hora). Si no hay cambios previos, devuelve la cantidad
                base del Menu 6. Se usa para saber a qué cantidad "revertir" cuando el cambio es temporal."""
                    anteriores = [
                        (inicio_cambio_flota_h(c, dur_turno_plan), c)
                        for c in st.session_state.plan_recursos_turno if c['recurso'] == recurso
                    ]
                    anteriores = [(t, c) for t, c in anteriores if t < inicio_h - 1e-9]
                    if not anteriores:
                        return int(cantidad_base)
                    t_max = max(t for t, _ in anteriores)
                    ultimo = [c for t, c in anteriores if abs(t - t_max) < 1e-9][-1]  # el último definido (igual que el motor)
                    return int(ultimo['cantidad_nueva'])

                def _registro_plan_limpio(fila):
                    """Quita las claves auxiliares/NaN que deja pandas al ir y volver de DataFrame."""
                    limpio = {k: v for k, v in fila.items() if k != '_inicio_h'}
                    for clave in ('hora', 'auto_revertido'):
                        if clave in limpio and pd.isna(limpio[clave]):
                            limpio.pop(clave)
                    return limpio

                with st.expander("Añadir cambio de cantidad de un recurso en un turno o en una hora"):
                    modo_tiempo_plan = st.radio(
                        "Ubicar el cambio por",
                        ["Turno", "Hora"],
                        horizontal=True,
                        key="plan_modo_tiempo",
                        help=f"'Turno': el cambio parte al inicio del turno indicado. 'Hora': parte en una hora exacta de la "
                             f"simulación, contada desde el inicio (hora 0). Con turnos de {dur_turno_plan:g} h, la hora "
                             f"{dur_turno_plan * 2 + dur_turno_plan / 2:g} cae a mitad del turno 3."
                    )
                    por_hora_plan = (modo_tiempo_plan == "Hora")

                    # Los 2 checkboxes viven FUERA del form (a diferencia de "Aplicar demora al
                    # cambio", que antes estaba adentro): dentro de un st.form, Streamlit no
                    # re-ejecuta el script al tildar un checkbox (solo al presionar el boton de
                    # submit), asi que el campo de horas de demora / turno de termino que dependen
                    # de estos checkboxes se quedaban "congelados" en su estado anterior y no se
                    # podian habilitar. Al vivir afuera, tildarlos S actualiza de inmediato (rerun
                    # de este fragment) el resto del formulario que se arma justo debajo.
                    col_chk1, col_chk2 = st.columns(2)
                    with col_chk1:
                        cambio_temporal = st.checkbox(
                            "Cambio temporal (revertir automáticamente al terminar)",
                            key="plan_cambio_temporal",
                            help="Ej: subir a 3 cuadrillas del turno 1 al turno 3 inclusive, y que desde "
                                 "el turno 4 vuelva a la cantidad que tenía antes. En modo 'Hora', el recurso "
                                 "vuelve a la cantidad anterior justo en la hora de término indicada."
                        )
                    with col_chk2:
                        con_demora = st.checkbox("Aplicar demora al cambio", key="plan_con_demora")

                    with st.form("form_plan_recurso_turno", clear_on_submit=True):
                        col_pr1, col_pr2, col_pr3, col_pr4 = st.columns(4)
                        with col_pr1:
                            recurso_plan = st.selectbox(
                                "Recurso",
                                sorted(recursos_nivel['Recurso'].unique()),
                                key="plan_recurso_select"
                            )
                        with col_pr2:
                            if por_hora_plan:
                                hora_plan = st.number_input(
                                    "Hora de inicio del cambio (h)",
                                    min_value=0.0,
                                    value=0.0,
                                    step=0.5,
                                    format="%.2f",
                                    key="plan_hora_input",
                                    help=f"Horas transcurridas desde el inicio de la simulación (hora 0). El programa actual "
                                         f"dura {horizonte_plan_h:g} h ({n_turnos_plan} turnos de {dur_turno_plan:g} h)."
                                )
                                turno_plan = None
                            else:
                                turno_plan = st.number_input(
                                    "Turno de inicio del cambio",
                                    min_value=1,
                                    value=1,
                                    step=1,
                                    key="plan_turno_input",
                                    help="Numeración de turnos igual a la usada en 'Número de Turnos del Programa' (1-based)."
                                )
                                hora_plan = None
                        with col_pr3:
                            cantidad_plan = st.number_input(
                                "Cantidad nueva a partir de ese turno" if not por_hora_plan else "Cantidad nueva a partir de esa hora",
                                min_value=0,
                                value=1,
                                step=1,
                                key="plan_cantidad_input"
                            )
                        with col_pr4:
                            demora_plan = st.number_input(
                                "Horas de demora en hacer efectivo el cambio",
                                min_value=0.0,
                                value=0.0,
                                step=0.5,
                                key="plan_demora_input",
                                disabled=not con_demora,
                                help="Ej: si sube la cantidad de Jumbos, el/los equipo(s) nuevo(s) solo estarán disponibles "
                                     "'demora_horas' después del inicio del turno (o de la hora) indicado. Si baja la cantidad, el/los "
                                     "equipo(s) retirado(s) dejan de estar disponibles 'demora_horas' después de ese inicio."
                            )

                        turno_fin_plan = None
                        hora_fin_plan = None
                        if cambio_temporal:
                            if por_hora_plan:
                                hora_fin_plan = st.number_input(
                                    "Hora de término (h) — desde esa hora, el recurso vuelve "
                                    "a la cantidad que tenía antes de este cambio",
                                    min_value=0.0,
                                    value=float(dur_turno_plan),
                                    step=0.5,
                                    format="%.2f",
                                    key="plan_hora_fin_input"
                                )
                            else:
                                turno_fin_plan = st.number_input(
                                    "Turno de término (inclusive) — desde el turno siguiente, el recurso vuelve "
                                    "a la cantidad que tenía antes de este cambio",
                                    min_value=int(turno_plan),
                                    value=int(turno_plan),
                                    step=1,
                                    key="plan_turno_fin_input"
                                )

                        submitted_plan = st.form_submit_button("Añadir cambio al plan", type="primary")
                        if submitted_plan:
                            if por_hora_plan:
                                inicio_h_plan = float(hora_plan)
                                fin_h_plan = float(hora_fin_plan) if cambio_temporal else None
                                turno_eq_plan = turno_de_hora(inicio_h_plan, dur_turno_plan)
                                error_plan = (
                                    "La hora de término debe ser posterior a la hora de inicio del cambio."
                                    if cambio_temporal and fin_h_plan <= inicio_h_plan else None
                                )
                            else:
                                inicio_h_plan = max(0.0, int(turno_plan) - 1.0) * dur_turno_plan
                                fin_h_plan = int(turno_fin_plan) * dur_turno_plan if cambio_temporal else None
                                turno_eq_plan = int(turno_plan)
                                error_plan = (
                                    "El turno de término no puede ser anterior al turno de inicio del cambio."
                                    if cambio_temporal and turno_fin_plan < turno_plan else None
                                )

                            if error_plan:
                                st.error(error_plan)
                            else:
                                cantidad_previa = _cantidad_vigente_antes_de(
                                    recurso_plan, inicio_h_plan,
                                    st.session_state.recursos_config.get(recurso_plan, {}).get('cantidad', 0)
                                )
                                entrada_plan = {
                                    'recurso': recurso_plan,
                                    'turno': turno_eq_plan,
                                    'cantidad_nueva': int(cantidad_plan),
                                    'demora_horas': float(demora_plan) if con_demora else 0.0
                                }
                                if por_hora_plan:
                                    entrada_plan['hora'] = round(inicio_h_plan, 4)
                                st.session_state.plan_recursos_turno.append(entrada_plan)

                                if por_hora_plan:
                                    mensaje = (f"Cambio añadido: **{recurso_plan}** → {int(cantidad_plan)} unidades desde la hora "
                                               f"{inicio_h_plan:g} (turno {turno_eq_plan})")
                                else:
                                    mensaje = f"Cambio añadido: **{recurso_plan}** → {int(cantidad_plan)} unidades desde el turno {int(turno_plan)}"

                                if cambio_temporal:
                                    revert_plan = {
                                        'recurso': recurso_plan,
                                        'turno': int(turno_fin_plan) + 1 if not por_hora_plan else turno_de_hora(fin_h_plan, dur_turno_plan),
                                        'cantidad_nueva': cantidad_previa,
                                        'demora_horas': 0.0,
                                        'auto_revertido': True
                                    }
                                    if por_hora_plan:
                                        revert_plan['hora'] = round(fin_h_plan, 4)
                                        mensaje += (f" hasta la hora {fin_h_plan:g}; desde esa hora "
                                                    f"vuelve a {cantidad_previa} unidades.")
                                    else:
                                        mensaje += (f" hasta el turno {int(turno_fin_plan)} (inclusive); "
                                                    f"desde el turno {int(turno_fin_plan) + 1} vuelve a {cantidad_previa} unidades.")
                                    st.session_state.plan_recursos_turno.append(revert_plan)
                                else:
                                    mensaje += "."
                                st.success(mensaje)
                                if inicio_h_plan >= horizonte_plan_h:
                                    st.warning(
                                        f"Ese cambio empieza en la hora {inicio_h_plan:g}, fuera del horizonte actual "
                                        f"({horizonte_plan_h:g} h): no afectará los resultados mientras el programa no sea más largo."
                                    )
                                reiniciar_resultados_simulacion("agregaste un cambio de flota (Orquestación de Flota por Turno u Hora)")

                if st.session_state.plan_recursos_turno:
                    st.markdown("#### Plan de cambios configurado")
                    df_plan = pd.DataFrame(st.session_state.plan_recursos_turno)
                    df_plan['_inicio_h'] = [inicio_cambio_flota_h(c, dur_turno_plan) for c in df_plan.to_dict('records')]
                    df_plan = df_plan.sort_values(['recurso', '_inicio_h'], kind='stable').reset_index(drop=True)
                    filas_plan = df_plan.to_dict('records')
                    es_auto = [bool(c.get('auto_revertido')) if not pd.isna(c.get('auto_revertido')) else False for c in filas_plan]
                    df_plan_display = pd.DataFrame({
                        'Recurso': df_plan['recurso'],
                        'Definido por': ['Hora' if _cambio_es_por_hora(c) else 'Turno' for c in filas_plan],
                        'Turno de Cambio': [
                            turno_de_hora(c['_inicio_h'], dur_turno_plan) if _cambio_es_por_hora(c) else int(c['turno'])
                            for c in filas_plan
                        ],
                        'Hora de Inicio (h)': df_plan['_inicio_h'].round(2),
                        'Cantidad Nueva': df_plan['cantidad_nueva'],
                        'Demora Aplicada (h)': df_plan['demora_horas'],
                        'Origen': ['Automático (fin de rango)' if a else 'Manual' for a in es_auto],
                    })
                    mostrar_tabla(df_plan_display, width='stretch', hide_index=True)

                    col_plan_del1, col_plan_del2 = st.columns([3, 1])
                    with col_plan_del1:
                        idx_a_borrar = st.multiselect(
                            "Seleccionar fila(s) del plan para eliminar (por índice mostrado arriba, empezando en 0)",
                            list(range(len(df_plan))),
                            key="plan_idx_borrar"
                        )
                    with col_plan_del2:
                        st.write("")
                        st.write("")
                        if st.button("Eliminar seleccionados", key="plan_borrar_btn", type="primary") and idx_a_borrar:
                            filas_restantes = [_registro_plan_limpio(f) for f in df_plan.drop(index=idx_a_borrar).to_dict('records')]
                            st.session_state.plan_recursos_turno = filas_restantes
                            reiniciar_resultados_simulacion("eliminaste cambios del plan de flota (Orquestación de Flota por Turno u Hora)")
                            st.rerun()

                    if st.button("Limpiar todo el plan de orquestación", key="plan_limpiar_btn", type="primary"):
                        st.session_state.plan_recursos_turno = []
                        reiniciar_resultados_simulacion("limpiaste el plan de orquestación de flota (Orquestación de Flota por Turno u Hora)")
                        st.rerun()
                else:
                    st.caption("Sin cambios de flota configurados.")

                st.session_state.cambios_recursos = st.session_state.plan_recursos_turno

                # ----------------------------------------------------
                # Gráfico: cantidad disponible de cada recurso a lo largo del programa
                # ----------------------------------------------------
                # Se arma desde el inicio con las cantidades base del Menu 6 (una línea plana
                # por recurso) y se actualiza solo, en vivo, con cada cambio que se añada/elimine
                # del plan. Cada escalón se dibuja en el instante en que el cambio se hace EFECTIVO
                # (inicio del turno u hora indicada + horas de demora), igual que lo aplica el motor.
                st.markdown("#### Cantidad de cada recurso disponible a lo largo del programa")
                recursos_lista_grafico = sorted(recursos_nivel['Recurso'].unique())

                if n_turnos_plan >= 1 and recursos_lista_grafico:
                    eje_x_flota = st.radio("Eje horizontal", ["Turno", "Hora"], horizontal=True, key="plan_eje_x_grafico")

                    def _demora_segura(ev):
                        d = ev.get('demora_horas')
                        return 0.0 if d is None or pd.isna(d) else float(d)

                    efectivos_por_recurso = {}
                    for recurso in recursos_lista_grafico:
                        evs = eventos_plan_flota_ordenados(st.session_state.plan_recursos_turno, recurso, dur_turno_plan)
                        efectivos_por_recurso[recurso] = sorted(
                            [(t + _demora_segura(ev), int(ev['cantidad_nueva'])) for t, ev in evs], key=lambda x: x[0]
                        )

                    # Malla común (inicio de cada turno + instante efectivo de cada cambio) para que el
                    # hover unificado muestre a todos los recursos en el mismo punto.
                    instantes = {round(i * dur_turno_plan, 6) for i in range(n_turnos_plan)} | {round(horizonte_plan_h, 6)}
                    for efectivos in efectivos_por_recurso.values():
                        instantes |= {round(t, 6) for t, _ in efectivos if t <= horizonte_plan_h}
                    malla = sorted(instantes)
                    eje_x_vals = [t / dur_turno_plan + 1.0 for t in malla] if eje_x_flota == "Turno" else malla

                    fig_flota_turno = go.Figure()
                    for recurso in recursos_lista_grafico:
                        cantidad_actual_base = int(st.session_state.recursos_config.get(recurso, {}).get('cantidad', 0))
                        serie_y = []
                        for t in malla:
                            cant = cantidad_actual_base
                            for t_ef, c_ef in efectivos_por_recurso[recurso]:
                                if t_ef <= t + 1e-9:
                                    cant = c_ef
                            serie_y.append(cant)

                        fig_flota_turno.add_trace(go.Scatter(
                            x=eje_x_vals,
                            y=serie_y,
                            mode='lines',
                            name=recurso,
                            line=dict(shape='hv', width=2.5),
                            hovertemplate=f"<b>{recurso}</b><br>{eje_x_flota} %{{x:.4~g}}<br>Cantidad: %{{y}}<extra></extra>"
                        ))

                    fig_flota_turno.update_layout(
                        xaxis_title="Turno del Programa" if eje_x_flota == "Turno" else "Hora de la simulación (h)",
                        yaxis_title="Cantidad Disponible",
                        height=400,
                        hovermode='x unified',
                        legend_title_text="Recurso",
                        margin=dict(l=10, r=10, t=30, b=10)
                    )
                    fig_flota_turno.update_yaxes(rangemode='tozero')
                    st.plotly_chart(fig_flota_turno, width='stretch')
                    ayuda_inline(
                        "Cada escalón aparece en el instante en que el cambio se hace efectivo: el inicio del turno (o la hora indicada) "
                        "más las horas de demora. En el eje 'Turno', un valor como 3.5 significa la mitad del turno 3; "
                        "el valor 1 es el inicio del turno 1.",
                        "Cómo leer el gráfico"
                    )
                else:
                    st.info("Defina el 'Número de Turnos del Programa' (Parámetros Generales) para ver este gráfico.")

        _menu_4b_orquestacion_flota(recursos_nivel)

        # ----------------------------------------------------
        # Modificar Distribuciones de Actividades
        # ----------------------------------------------------

        # ======================================================================
        # #####Menu 8 — Modificar Distribuciones de Actividades#####
        # ======================================================================
        with seccion_menu('8', 'Modificar Distribuciones de Actividades'):
            with st.expander("Distribuciones disponibles y sus parámetros (scipy.stats)"):
                usos_referencia = {
                    'Cte': 'Valor constante', 'fisk': 'Perforación', 'kstwobign': 'Transporte/Acarreo',
                    'rayleigh': 'Empotramiento', 'foldcauchy': 'Sostenimiento, Shotcrete',
                    'foldnorm': 'Perforación Fin', 'ncx2': 'Instalación',
                    'burr': 'colas pesadas, uso general', 'loglaplace': 'picos con colas asimétricas',
                    'maxwell': 'velocidades, siempre positiva', 'nakagami': 'variabilidad de señal/desempeño',
                    'normal': 'estándar', 'weibull': 'estándar', 'gamma': 'estándar', 'lognormal': 'estándar',
                }
                for nombre_d, info_d in DISTRIBUTION_PARAM_INFO.items():
                    params_str = ", ".join(p['label'] for p in info_d['parametros'])
                    uso = usos_referencia.get(nombre_d, '')
                    st.markdown(f"- **{nombre_d}** ({uso}): {params_str}")

            actividades_lista = actividades_nivel[actividades_nivel['Actividad'].notna() & (actividades_nivel['Es_Personalizada'] == False)].copy()

            if 'actividades_seleccionadas' not in st.session_state:
                st.session_state.actividades_seleccionadas = []

            # ---- Filtros: por Actividad y/o por Distribución ----
            st.markdown("#### Filtrar Actividades (por Actividad y/o Distribución):")
            actividades_presentes = sorted(actividades_lista['Actividad'].dropna().astype(str).unique().tolist())
            distribuciones_presentes = sorted(actividades_lista['Distribucion'].dropna().unique().tolist())

            col_f1, col_f2 = st.columns(2)
            with col_f1:
                act_filtro_sel = st.multiselect(
                    "Filtrar por Actividad (vacío = todas)",
                    actividades_presentes,
                    default=[],
                    key="act_filtro_multiselect"
                )
            with col_f2:
                dist_filtro_sel = st.multiselect(
                    "Filtrar por Distribución (vacío = todas)",
                    distribuciones_presentes,
                    default=[],
                    key="dist_filtro_multiselect"
                )

            actividades_filtradas = actividades_lista.copy()
            if act_filtro_sel:
                actividades_filtradas = actividades_filtradas[actividades_filtradas['Actividad'].astype(str).isin(act_filtro_sel)]
            if dist_filtro_sel:
                actividades_filtradas = actividades_filtradas[actividades_filtradas['Distribucion'].isin(dist_filtro_sel)]

            hay_filtro_activo = bool(act_filtro_sel) or bool(dist_filtro_sel)

            # Actividades visibles segun el filtro (la tabla y el selector individual usan este mismo conjunto)
            indices_visibles = (actividades_filtradas if hay_filtro_activo else actividades_lista).index.tolist()
            CLAVE_SEL_MULTI = "actividades_sel_multiselect"

            def _sincronizar_sel_multi():
                st.session_state[CLAVE_SEL_MULTI] = [
                    i for i in st.session_state.actividades_seleccionadas if i in indices_visibles]

            # Botones de selección masiva: sobre TODAS las actividades, o solo sobre el tramo filtrado
            col_sel1, col_sel2, col_sel3, col_sel4 = st.columns([1, 1, 1.3, 2.7])
            with col_sel1:
                if st.button("Seleccionar Todo", type="primary", width='stretch'):
                    st.session_state.actividades_seleccionadas = actividades_lista.index.tolist()
                    _sincronizar_sel_multi()
                    st.rerun()
            with col_sel2:
                if st.button("Deseleccionar Todo", type="primary", width='stretch'):
                    st.session_state.actividades_seleccionadas = []
                    _sincronizar_sel_multi()
                    st.rerun()
            with col_sel3:
                if st.button("Seleccionar Filtrados", type="primary", width='stretch', disabled=not hay_filtro_activo):
                    st.session_state.actividades_seleccionadas = actividades_filtradas.index.tolist()
                    _sincronizar_sel_multi()
                    st.rerun()

            tabla_a_mostrar = actividades_filtradas if hay_filtro_activo else actividades_lista
            actividades_para_mostrar = tabla_a_mostrar.copy()
            actividades_para_mostrar['Seleccionado'] = actividades_para_mostrar.index.isin(st.session_state.actividades_seleccionadas)

            if hay_filtro_activo:
                st.markdown(f"**Actividades Filtradas ({len(tabla_a_mostrar)} de {len(actividades_lista)} originales):**")
            else:
                st.markdown(f"**Actividades Originales ({len(actividades_lista)}):**")
            df_display = actividades_para_mostrar[['Actividad', 'Recurso', 'Secuencia', 'Distribucion', 'Tiempo', 'Seleccionado']].copy()

            def _num_corto(v):
                try:
                    return f"{float(v):.3f}".rstrip('0').rstrip('.')
                except (TypeError, ValueError):
                    return str(v)

            def _tiempo_legible(v):
                """'Tiempo' mezcla constantes (numeros) con parametros (dict). Siempre se muestra como texto
                corto (mean=1, std=0.1); si se deja numerico, al filtrar solo 'Cte' st.data_editor revienta."""
                if isinstance(v, dict):
                    partes = [f"{k}={_num_corto(x)}" for k, x in v.items() if not str(k).startswith('_trunc')]
                    txt = ", ".join(partes)
                    if '_trunc_lo' in v and '_trunc_hi' in v:
                        txt += f"  | rango [{_num_corto(v['_trunc_lo'])}, {_num_corto(v['_trunc_hi'])}]"
                    return txt
                if v is None or pd.isna(v) is True:
                    return "—"
                return _num_corto(v)

            df_display['Tiempo'] = df_display['Tiempo'].map(_tiempo_legible)
            df_display['Recurso'] = df_display['Recurso'].map(
                lambda v: "—" if v is None or pd.isna(v) is True or str(v).strip() in ('', 'None', 'nan') else str(v))
            df_display['Estado'] = [
                "Modificada" if i in st.session_state.actividades_modificadas else "Original"
                for i in df_display.index
            ]
            df_display = df_display[['Actividad', 'Recurso', 'Secuencia', 'Distribucion', 'Tiempo', 'Estado', 'Seleccionado']]

            # Selección individual: el selector reemplaza a la casilla de la tabla, para que la tabla pueda
            # seguir el formato estándar del programa (encabezado y contenido centrados).
            def _etiqueta_fila(i):
                fila = actividades_lista.loc[i]
                rec = fila['Recurso']
                rec = "" if rec is None or pd.isna(rec) is True or str(rec).strip() in ('', 'None', 'nan') else str(rec)
                seq = _num_corto(fila['Secuencia'])
                return f"{fila['Actividad']} · {rec} · Sec. {seq}" if rec else f"{fila['Actividad']} · Sec. {seq}"

            contenedor_tabla = st.container()   # la tabla se dibuja arriba, ya con la selección actualizada
            firma_visibles = tuple(indices_visibles)
            if CLAVE_SEL_MULTI not in st.session_state or st.session_state.get('_sel_visibles_firma') != firma_visibles:
                st.session_state['_sel_visibles_firma'] = firma_visibles
                _sincronizar_sel_multi()
            seleccion_en_vista = st.multiselect(
                "Actividades a modificar (selección individual)",
                options=indices_visibles,
                format_func=_etiqueta_fila,
                key=CLAVE_SEL_MULTI,
                help="Marca o desmarca actividades una a una. También puedes usar los botones de arriba."
            )

            # Las filas visibles (posiblemente filtradas) actualizan su estado de selección;
            # las filas fuera del filtro actual conservan la selección que ya tenían.
            seleccion_previa = set(st.session_state.actividades_seleccionadas)
            indices_fuera_de_vista = seleccion_previa - set(indices_visibles)
            st.session_state.actividades_seleccionadas = list(indices_fuera_de_vista | set(seleccion_en_vista))

            tabla_vista = df_display.copy()
            tabla_vista['Seleccionado'] = ["✔ Sí" if i in st.session_state.actividades_seleccionadas else "—"
                                           for i in tabla_vista.index]
            tabla_vista = tabla_vista.rename(columns={'Distribucion': 'Distribución', 'Tiempo': 'Tiempo (parámetros)'})
            with contenedor_tabla:
                mostrar_tabla(tabla_vista, hide_index=True, width='stretch')

            if st.session_state.actividades_seleccionadas:
                st.markdown("#### Configuración de la Modificación:")
            
                sel_idx = [i for i in st.session_state.actividades_seleccionadas if i in actividades_lista.index]
                actividad_ejemplo = str(actividades_lista.loc[sel_idx[0]]['Actividad']) if sel_idx else "—"

                def _etiqueta_act(i):
                    fila = actividades_lista.loc[i]
                    rec = fila['Recurso']
                    rec = "" if rec is None or pd.isna(rec) is True or str(rec).strip() in ('', 'None', 'nan') else str(rec)
                    return f"{fila['Actividad']} ({rec})" if rec else f"{fila['Actividad']}"

                etiquetas_sel = list(dict.fromkeys(_etiqueta_act(i) for i in sel_idx))
                ya_config = [i for i in sel_idx if i in st.session_state.actividades_modificadas]
                etiquetas_cfg = list(dict.fromkeys(_etiqueta_act(i) for i in ya_config))
                total_cfg = len(st.session_state.actividades_modificadas)

                msg_cfg = f"Configurando modificación para {len(sel_idx)} actividades (Ej. {actividad_ejemplo})."
                st.info(msg_cfg)
                detalle_cfg = "Seleccionadas: " + ", ".join(etiquetas_sel) + "."
                detalle_cfg += (f" Ya configuradas antes ({len(ya_config)}): " + ", ".join(etiquetas_cfg) + "."
                                if ya_config else " Ninguna de las seleccionadas ha sido configurada todavía.")
                detalle_cfg += f" Total de actividades modificadas en el programa: {total_cfg}."
                st.caption(detalle_cfg)
                log_evento('INFO', f"{msg_cfg} {detalle_cfg}", dedupe=True)

                col_mod1, col_mod2 = st.columns(2)
                with col_mod1:
                    nueva_dist = st.selectbox(
                        "Nueva Distribución de Tiempo",
                        ['Cte', 'normal', 'weibull', 'gamma', 'lognormal', 'fisk', 'kstwobign',
                         'rayleigh', 'foldcauchy', 'foldnorm', 'ncx2',
                         'burr', 'loglaplace', 'maxwell', 'nakagami'],
                        key="nueva_dist"
                    )

                with col_mod2:
                    # Los inputs de cada parametro se generan automaticamente a partir de
                    # DISTRIBUTION_PARAM_INFO, usando los nombres reales de scipy.stats
                    # (ej. gamma usa 'a' y 'scale', no "parametro 1 y 2").
                    info_dist = DISTRIBUTION_PARAM_INFO[nueva_dist]
                    valores_ingresados = {}
                    for p in info_dist['parametros']:
                        valores_ingresados[p['key']] = st.number_input(
                            p['label'],
                            min_value=p.get('min'),
                            value=p['default'],
                            step=0.01 if p.get('min') == 0.0001 else 0.1,
                            key=f"param_{nueva_dist}_{p['key']}"
                        )

                    if nueva_dist == 'Cte':
                        nuevos_params = valores_ingresados['valor']
                    else:
                        nuevos_params = valores_ingresados

                # ---- Vista previa grafica de la PDF (punto 2: validar visualmente el cambio) ----
                st.markdown("##### Vista Previa de la Distribución (PDF):")
                if nueva_dist == 'Cte':
                    st.info(f"Distribución constante: siempre entrega **{nuevos_params:.3f} h**. No aplica curva de densidad.")
                else:
                    try:
                        # Solo se arma el kwargs de la distribucion ACTUALMENTE seleccionada.
                        # (Antes se armaba un diccionario con las 14 distribuciones de una vez,
                        # lo que evaluaba nuevos_params['mean'] aunque la distribucion elegida
                        # fuera 'burr', reventando con KeyError: 'mean'.)
                        dist_kwargs_por_nombre = {
                            'normal': lambda: {'loc': nuevos_params['mean'], 'scale': nuevos_params['std']},
                            'weibull': lambda: {'c': nuevos_params['c'], 'loc': nuevos_params['loc'], 'scale': nuevos_params['scale']},
                            'gamma': lambda: {'a': nuevos_params['a'], 'loc': nuevos_params['loc'], 'scale': nuevos_params['scale']},
                            'lognormal': lambda: {'s': nuevos_params['s'], 'loc': nuevos_params['loc'], 'scale': nuevos_params['scale']},
                            'fisk': lambda: {'c': nuevos_params['c'], 'loc': nuevos_params['loc'], 'scale': nuevos_params['scale']},
                            'kstwobign': lambda: {'loc': nuevos_params['loc'], 'scale': nuevos_params['scale']},
                            'rayleigh': lambda: {'loc': nuevos_params['loc'], 'scale': nuevos_params['scale']},
                            'foldcauchy': lambda: {'c': nuevos_params['c'], 'loc': nuevos_params['loc'], 'scale': nuevos_params['scale']},
                            'foldnorm': lambda: {'c': nuevos_params['c'], 'loc': nuevos_params['loc'], 'scale': nuevos_params['scale']},
                            'ncx2': lambda: {'df': nuevos_params['df'], 'nc': nuevos_params['nc'], 'loc': nuevos_params['loc'], 'scale': nuevos_params['scale']},
                            'burr': lambda: {'c': nuevos_params['c'], 'd': nuevos_params['d'], 'loc': nuevos_params['loc'], 'scale': nuevos_params['scale']},
                            'loglaplace': lambda: {'c': nuevos_params['c'], 'loc': nuevos_params['loc'], 'scale': nuevos_params['scale']},
                            'maxwell': lambda: {'loc': nuevos_params['loc'], 'scale': nuevos_params['scale']},
                            'nakagami': lambda: {'nu': nuevos_params['nu'], 'loc': nuevos_params['loc'], 'scale': nuevos_params['scale']},
                        }
                        dist_obj_por_nombre = {
                            'normal': stats.norm, 'weibull': stats.weibull_min, 'gamma': stats.gamma,
                            'lognormal': stats.lognorm, 'fisk': stats.fisk, 'kstwobign': stats.kstwobign,
                            'rayleigh': stats.rayleigh, 'foldcauchy': stats.foldcauchy, 'foldnorm': stats.foldnorm,
                            'ncx2': stats.ncx2, 'burr': stats.burr, 'loglaplace': stats.loglaplace,
                            'maxwell': stats.maxwell, 'nakagami': stats.nakagami,
                        }
                        dist_obj = dist_obj_por_nombre[nueva_dist]
                        dist_kwargs = dist_kwargs_por_nombre[nueva_dist]()

                        # Rango de x centrado en la forma de la distribucion elegida, usando
                        # los percentiles 0.1% y 99.9% de la propia distribucion (ppf) para
                        # no adivinar limites a mano.
                        x_lo = dist_obj.ppf(0.001, **dist_kwargs)
                        x_hi = dist_obj.ppf(0.999, **dist_kwargs)
                        if not np.isfinite(x_lo):
                            x_lo = 0.0
                        if not np.isfinite(x_hi):
                            x_hi = x_lo + 10.0
                        if x_hi <= x_lo:
                            x_hi = x_lo + 1.0
                        x_vals_preview = np.linspace(x_lo, x_hi, 400)
                        y_vals_preview = calcular_pdf_teorica(nuevos_params, nueva_dist, x_vals_preview)

                        fig_preview = go.Figure()
                        fig_preview.add_trace(go.Scatter(
                            x=x_vals_preview, y=y_vals_preview,
                            mode='lines', fill='tozeroy',
                            line=dict(color='#1f77b4', width=2),
                            name=nueva_dist
                        ))

                        # ---- Filtro de rango en X sobre la propia curva (punto 3: "que pasa
                        # si solo miro los tiempos mayores/menores/entre tales valores", para
                        # fines de optimizacion). Sombrea el area correspondiente y calcula su
                        # probabilidad exacta con la CDF, sin aproximar con el histograma. ----
                        col_area1, col_area2 = st.columns([1, 2])
                        with col_area1:
                            modo_area = st.selectbox(
                                "Área bajo la curva a resaltar",
                                ["Ninguna", "Mayor a (>)", "Menor a (<)", "Entre (rango)"],
                                key=f"modo_area_{nueva_dist}"
                            )
                        with col_area2:
                            x_area_lo, x_area_hi = None, None
                            if modo_area == "Mayor a (>)":
                                x_area_lo = st.slider("Valor de corte (h)", float(x_lo), float(x_hi),
                                                       value=float((x_lo + x_hi) / 2), key=f"corte_mayor_{nueva_dist}")
                                x_area_hi = x_hi
                            elif modo_area == "Menor a (<)":
                                x_area_hi = st.slider("Valor de corte (h)", float(x_lo), float(x_hi),
                                                       value=float((x_lo + x_hi) / 2), key=f"corte_menor_{nueva_dist}")
                                x_area_lo = x_lo
                            elif modo_area == "Entre (rango)":
                                x_area_lo, x_area_hi = st.slider(
                                    "Rango (h)", float(x_lo), float(x_hi),
                                    value=(float(x_lo), float(x_hi)), key=f"corte_rango_{nueva_dist}"
                                )

                        if modo_area != "Ninguna" and x_area_lo is not None:
                            mascara_area = (x_vals_preview >= x_area_lo) & (x_vals_preview <= x_area_hi)
                            fig_preview.add_trace(go.Scatter(
                                x=x_vals_preview[mascara_area], y=y_vals_preview[mascara_area],
                                mode='lines', fill='tozeroy',
                                line=dict(width=0), fillcolor='rgba(255,99,71,0.55)',
                                name='Área seleccionada'
                            ))
                            prob_area = dist_obj.cdf(x_area_hi, **dist_kwargs) - dist_obj.cdf(x_area_lo, **dist_kwargs)
                            st.caption(f"Probabilidad de que el tiempo caiga en [{x_area_lo:.3f}, {x_area_hi:.3f}] h: **{prob_area*100:.2f}%** del área total.")

                        fig_preview.update_layout(
                            height=280,
                            margin=dict(l=40, r=20, t=20, b=40),
                            xaxis_title="Tiempo (h)",
                            yaxis_title="Densidad de probabilidad",
                            showlegend=False
                        )
                        st.plotly_chart(fig_preview, width='stretch', key="preview_pdf_nueva_dist")
                    except Exception as e:
                        st.warning(f"No fue posible graficar la vista previa con los parámetros actuales: {e}")

                if st.button("Aplicar Modificación de Distribución", type="primary"):
                    if nuevos_params is not None:
                        # Si el usuario dejo seleccionada un area (Mayor a / Menor a / Entre)
                        # en la vista previa, ese rango se guarda como truncamiento real de la
                        # distribucion: '_trunc_lo'/'_trunc_hi' dentro de los propios parametros.
                        # generar_tiempos_batch() y calcular_pdf_teorica() ya saben leer estas
                        # claves e invertir la CDF restringida a ese tramo, asi que los tiempos
                        # que salgan de la simulacion quedan garantizados dentro del rango
                        # (nada de valores en 0, 1 o fuera de [2, 3] si se filtro "Entre 2 y 3").
                        # 'Cte' no es un dict (es un float) y tampoco tiene sentido truncarla:
                        # una constante ya es un unico valor fijo, sin distribucion que recortar.
                        if isinstance(nuevos_params, dict):
                            modo_area_actual = st.session_state.get(f"modo_area_{nueva_dist}", "Ninguna")
                            # x_lo/x_hi solo existen como variable local si la vista previa se
                            # armo sin excepciones; si algo fallo arriba, se cae a los propios
                            # sliders (que ya traen su rango real guardado por Streamlit).
                            x_lo_local = locals().get('x_lo')
                            x_hi_local = locals().get('x_hi')
                            if modo_area_actual == "Mayor a (>)":
                                corte_mayor = st.session_state.get(f"corte_mayor_{nueva_dist}")
                                if corte_mayor is not None and x_hi_local is not None:
                                    nuevos_params['_trunc_lo'] = float(corte_mayor)
                                    nuevos_params['_trunc_hi'] = float(x_hi_local)
                            elif modo_area_actual == "Menor a (<)":
                                corte_menor = st.session_state.get(f"corte_menor_{nueva_dist}")
                                if corte_menor is not None and x_lo_local is not None:
                                    nuevos_params['_trunc_lo'] = float(x_lo_local)
                                    nuevos_params['_trunc_hi'] = float(corte_menor)
                            elif modo_area_actual == "Entre (rango)":
                                rango_actual = st.session_state.get(f"corte_rango_{nueva_dist}")
                                if rango_actual is not None:
                                    nuevos_params['_trunc_lo'] = float(rango_actual[0])
                                    nuevos_params['_trunc_hi'] = float(rango_actual[1])
                            else:
                                nuevos_params.pop('_trunc_lo', None)
                                nuevos_params.pop('_trunc_hi', None)

                        for idx in st.session_state.actividades_seleccionadas:
                            st.session_state.actividades_modificadas[idx] = {
                                'Distribucion': nueva_dist,
                                'Tiempo': nuevos_params
                            }
                        try:
                            _nombres_aplic = list(dict.fromkeys(
                                _etiqueta_act(i) for i in st.session_state.actividades_seleccionadas if i in actividades_lista.index))
                            log_evento('SUCCESS',
                                       f"Modificación de distribución aplicada: '{nueva_dist}' a "
                                       f"{len(st.session_state.actividades_seleccionadas)} actividades ({', '.join(_nombres_aplic)}). "
                                       f"Total de actividades modificadas en el programa: {len(st.session_state.actividades_modificadas)}.")
                        except Exception:
                            pass
                        # El cambio se aplica de inmediato: se fuerza un rerun para que el
                        # bloque "2. Seleccionar Nivel" (que lee actividades_modificadas en
                        # cada ejecucion) refresque la tabla ya con la nueva distribucion,
                        # sin que el usuario tenga que volver a tocar el selector de Nivel.
                        if isinstance(nuevos_params, dict) and '_trunc_lo' in nuevos_params:
                            st.success(
                                f"Distribución '{nueva_dist}' aplicada a {len(st.session_state.actividades_seleccionadas)} "
                                f"actividades, truncada al rango [{nuevos_params['_trunc_lo']:.3f}, {nuevos_params['_trunc_hi']:.3f}] h."
                            )
                        else:
                            st.success(f"Distribución '{nueva_dist}' aplicada a {len(st.session_state.actividades_seleccionadas)} actividades.")
                        reiniciar_resultados_simulacion("modificaste la distribución de tiempo de una o más actividades (Modificar Distribuciones de Actividades)")
                        st.rerun()
                    else:
                        st.error("Error: Los parámetros de distribución no son válidos.")

            # frentes_disponibles se calcula una sola vez aqui y se reutiliza en los Menus
            # 9, 10 y 11 (antes se recalculaba dentro del Menu 9 y se leia "prestado" desde ahi).
            frentes_disponibles = frentes_nivel['Frentes'].unique()

        # ----------------------------------------------------
        # Agregar Actividades Personalizadas
        # ----------------------------------------------------
        # Todo el Menu 9 vive dentro de un @st.fragment, igual que los Menus 5, 7 y 16:
        # el form ya evitaba rerun tecla por tecla, pero "Añadir Actividad" o "Limpiar
        # Actividades Personalizadas" disparaban un rerun de la app COMPLETA. Encerrado en
        # el fragment, esos botones solo vuelven a ejecutar este bloque.
        #
        # Nota: se dejo de usar st.form aqui. Un form congela sus widgets hasta el submit,
        # por lo que la vista previa de la PDF y el filtro de area (que necesitan
        # reaccionar en cada cambio de parametro, igual que en el Menu 5) no podian
        # actualizarse en vivo dentro de uno. Al vivir sueltos dentro del fragment, cada
        # widget dispara solo un rerun de este bloque (no de la app completa) y la
        # vista previa queda sincronizada con lo que el usuario esta tocando.
        @st.fragment
        def _menu_6_actividades_personalizadas(recursos_nivel, frentes_disponibles, nivel_seleccionado, actividades_nivel):

            # ======================================================================
            # #####Menu 9 — Agregar Actividades Personalizadas#####
            # ======================================================================
            with seccion_menu('9', 'Agregar Actividades Personalizadas'):

                distribuciones_disponibles = ['Cte', 'normal', 'weibull', 'gamma', 'lognormal', 'fisk', 'kstwobign',
                                              'rayleigh', 'foldcauchy', 'foldnorm', 'ncx2',
                                              'burr', 'loglaplace', 'maxwell', 'nakagami']

                recursos_disponibles_act = ['N/A'] + list(recursos_nivel['Recurso'].unique())

                # El campo de nombre se limpia aqui, ANTES de instanciar el widget.
                # Streamlit no permite modificar st.session_state[key] una vez que
                # el widget con esa key ya fue creado en el run actual, por eso el
                # boton "Añadir Actividad" solo deja una bandera (_reset_nueva_act_nombre)
                # y es este bloque, ejecutado antes del st.text_input, el que la aplica.
                if st.session_state.get("_reset_nueva_act_nombre", False):
                    st.session_state["nueva_act_nombre"] = ""
                    st.session_state["_reset_nueva_act_nombre"] = False

                with st.expander("Añadir Nueva Actividad Personalizada"):
                    col_n1, col_n2, col_n3 = st.columns(3)
                    with col_n1:
                        nombre_act = st.text_input("Nombre de la Actividad", key="nueva_act_nombre")
                        recurso_act = st.selectbox(
                            "Recurso Requerido (N/A = Sin Recurso)",
                            recursos_disponibles_act,
                            key="nueva_act_recurso"
                        )
                    with col_n2:
                        distribucion_act = st.selectbox(
                            "Tipo de Distribución",
                            distribuciones_disponibles,
                            key="nueva_act_distribucion"
                        )

                        frentes_seleccionados_act = st.multiselect(
                            "Túneles (Frentes) Aplicables",
                            frentes_disponibles,
                            default=frentes_disponibles,
                            key="nueva_act_frentes"
                        )

                    with col_n3:
                        st.markdown("#### Parámetros de Tiempo (Horas)")
                        # Misma fuente unica que el Menu 5 (DISTRIBUTION_PARAM_INFO): asi los
                        # inputs de cada distribucion se generan con los nombres reales de
                        # scipy.stats en vez de reescribir 14 ramas "a mano".
                        info_dist_act = DISTRIBUTION_PARAM_INFO[distribucion_act]
                        valores_ingresados_act = {}
                        for p in info_dist_act['parametros']:
                            valores_ingresados_act[p['key']] = st.number_input(
                                p['label'],
                                min_value=p.get('min'),
                                value=p['default'],
                                step=0.01 if p.get('min') == 0.0001 else 0.1,
                                key=f"param_add_{distribucion_act}_{p['key']}"
                            )

                        if distribucion_act == 'Cte':
                            params_act = valores_ingresados_act['valor']
                        else:
                            params_act = valores_ingresados_act

                    # ---- Posición dentro de la secuencia del nivel ----
                    st.markdown("##### Posición en la secuencia")
                    base_orden = st.session_state.get('_act_base_nivel')
                    max_base = int(pd.to_numeric(base_orden['Secuencia'], errors='coerce').max()) \
                        if base_orden is not None and len(base_orden) else 0
                    nombre_limpio_act = (nombre_act or "").strip()
                    otras_pers = [a for a in st.session_state.actividades_adicionales
                                  if a['Actividad'] != nombre_limpio_act]
                    total_pasos = max_base + len(otras_pers)
                    col_pos1, col_pos2 = st.columns([1, 2])
                    with col_pos1:
                        modo_pos_act = st.radio(
                            "¿Dónde va?", ["Al final", "Elegir posición"],
                            horizontal=True, key="nueva_act_modo_pos")
                        posicion_act = None
                        if modo_pos_act == "Elegir posición":
                            posicion_act = int(st.number_input(
                                f"Posición (1 a {total_pasos + 1})", min_value=1,
                                max_value=total_pasos + 1, value=total_pasos + 1,
                                step=1, key="nueva_act_posicion",
                                help="Las actividades que ya están en esa posición (o después) se corren una hacia abajo. La posición es la del listado final: con 30 actividades, dos nuevas en la 16 y la 22 quedan exactamente en la 16 y la 22."))
                    with col_pos2:
                        pass
                    if base_orden is not None and len(base_orden):
                        with st.expander("Ver cómo queda el orden de actividades del nivel"):
                            _hipotetica = otras_pers + [{
                                'Actividad': nombre_limpio_act or "(nueva actividad)",
                                'Recurso': recurso_act if recurso_act != 'N/A' else np.nan,
                                'Posicion': posicion_act,
                            }]
                            _base_prev, _nuevas_prev = aplicar_secuencias_personalizadas(base_orden, _hipotetica)
                            _orden = pd.concat([
                                _base_prev.assign(Origen='Existente'),
                                pd.DataFrame(_nuevas_prev)[['Secuencia', 'Actividad', 'Recurso']].assign(Origen='Nueva'),
                            ], ignore_index=True).sort_values(['Secuencia', 'Origen'], kind='stable')
                            mostrar_tabla(_orden[['Secuencia', 'Actividad', 'Recurso', 'Origen']],
                                          width='stretch', hide_index=True)

                    # ---- Vista previa de la PDF + filtro de area, igual que en el Menu 5 ----
                    x_lo_act, x_hi_act = None, None
                    if distribucion_act == 'Cte':
                        st.info(f"Distribución constante: siempre entrega **{params_act:.3f} h**. No aplica curva de densidad.")
                    else:
                        try:
                            dist_obj_act = DIST_OBJ_SCIPY[distribucion_act]
                            dist_kwargs_act = _kwargs_scipy_desde_params(params_act, distribucion_act)

                            x_lo_act = dist_obj_act.ppf(0.001, **dist_kwargs_act)
                            x_hi_act = dist_obj_act.ppf(0.999, **dist_kwargs_act)
                            if not np.isfinite(x_lo_act):
                                x_lo_act = 0.0
                            if not np.isfinite(x_hi_act):
                                x_hi_act = x_lo_act + 10.0
                            if x_hi_act <= x_lo_act:
                                x_hi_act = x_lo_act + 1.0
                            x_vals_act = np.linspace(x_lo_act, x_hi_act, 400)
                            y_vals_act = calcular_pdf_teorica(params_act, distribucion_act, x_vals_act)

                            fig_preview_act = go.Figure()
                            fig_preview_act.add_trace(go.Scatter(
                                x=x_vals_act, y=y_vals_act,
                                mode='lines', fill='tozeroy',
                                line=dict(color='#1f77b4', width=2),
                                name=distribucion_act
                            ))

                            st.markdown("##### Vista Previa de la Distribución (PDF):")
                            col_area_act1, col_area_act2 = st.columns([1, 2])
                            with col_area_act1:
                                modo_area_act = st.selectbox(
                                    "Área bajo la curva a resaltar",
                                    ["Ninguna", "Mayor a (>)", "Menor a (<)", "Entre (rango)"],
                                    key=f"modo_area_add_{distribucion_act}"
                                )
                            with col_area_act2:
                                x_area_lo_act, x_area_hi_act = None, None
                                if modo_area_act == "Mayor a (>)":
                                    x_area_lo_act = st.slider("Valor de corte (h)", float(x_lo_act), float(x_hi_act),
                                                               value=float((x_lo_act + x_hi_act) / 2), key=f"corte_mayor_add_{distribucion_act}")
                                    x_area_hi_act = x_hi_act
                                elif modo_area_act == "Menor a (<)":
                                    x_area_hi_act = st.slider("Valor de corte (h)", float(x_lo_act), float(x_hi_act),
                                                               value=float((x_lo_act + x_hi_act) / 2), key=f"corte_menor_add_{distribucion_act}")
                                    x_area_lo_act = x_lo_act
                                elif modo_area_act == "Entre (rango)":
                                    x_area_lo_act, x_area_hi_act = st.slider(
                                        "Rango (h)", float(x_lo_act), float(x_hi_act),
                                        value=(float(x_lo_act), float(x_hi_act)), key=f"corte_rango_add_{distribucion_act}"
                                    )

                            if modo_area_act != "Ninguna" and x_area_lo_act is not None:
                                mascara_area_act = (x_vals_act >= x_area_lo_act) & (x_vals_act <= x_area_hi_act)
                                fig_preview_act.add_trace(go.Scatter(
                                    x=x_vals_act[mascara_area_act], y=y_vals_act[mascara_area_act],
                                    mode='lines', fill='tozeroy',
                                    line=dict(width=0), fillcolor='rgba(255,99,71,0.55)',
                                    name='Área seleccionada'
                                ))
                                prob_area_act = dist_obj_act.cdf(x_area_hi_act, **dist_kwargs_act) - dist_obj_act.cdf(x_area_lo_act, **dist_kwargs_act)
                                st.caption(f"Probabilidad de que el tiempo caiga en [{x_area_lo_act:.3f}, {x_area_hi_act:.3f}] h: **{prob_area_act*100:.2f}%** del área total.")

                            fig_preview_act.update_layout(
                                height=280,
                                margin=dict(l=40, r=20, t=20, b=40),
                                xaxis_title="Tiempo (h)",
                                yaxis_title="Densidad de probabilidad",
                                showlegend=False
                            )
                            st.plotly_chart(fig_preview_act, width='stretch', key="preview_pdf_nueva_actividad")
                        except Exception as e:
                            st.warning(f"No fue posible graficar la vista previa con los parámetros actuales: {e}")

                    if st.button("Añadir Actividad", type="primary", key="btn_add_actividad"):
                        if posicion_act and any(a.get('Posicion') == posicion_act for a in otras_pers):
                            st.error(f"La posición {posicion_act} ya la ocupa otra actividad personalizada. Elige otra posición.")
                        elif nombre_act and frentes_seleccionados_act and params_act is not None:
                            # Igual que en el Menu 5: si se dejo seleccionada un area, ese
                            # rango se guarda como truncamiento real ('_trunc_lo'/'_trunc_hi')
                            # dentro de los propios parametros, no solo como indicador visual.
                            if isinstance(params_act, dict):
                                modo_area_actual_act = st.session_state.get(f"modo_area_add_{distribucion_act}", "Ninguna")
                                if modo_area_actual_act == "Mayor a (>)":
                                    corte_mayor_act = st.session_state.get(f"corte_mayor_add_{distribucion_act}")
                                    if corte_mayor_act is not None and x_hi_act is not None:
                                        params_act['_trunc_lo'] = float(corte_mayor_act)
                                        params_act['_trunc_hi'] = float(x_hi_act)
                                elif modo_area_actual_act == "Menor a (<)":
                                    corte_menor_act = st.session_state.get(f"corte_menor_add_{distribucion_act}")
                                    if corte_menor_act is not None and x_lo_act is not None:
                                        params_act['_trunc_lo'] = float(x_lo_act)
                                        params_act['_trunc_hi'] = float(corte_menor_act)
                                elif modo_area_actual_act == "Entre (rango)":
                                    rango_actual_act = st.session_state.get(f"corte_rango_add_{distribucion_act}")
                                    if rango_actual_act is not None:
                                        params_act['_trunc_lo'] = float(rango_actual_act[0])
                                        params_act['_trunc_hi'] = float(rango_actual_act[1])
                                else:
                                    params_act.pop('_trunc_lo', None)
                                    params_act.pop('_trunc_hi', None)

                            nueva_actividad_data = {
                                'Nivel': nivel_seleccionado,
                                'Secuencia': posicion_act if posicion_act else total_pasos + 1,
                                'Posicion': posicion_act,
                                'Actividad': nombre_act.strip(),
                                'Recurso': recurso_act if recurso_act != 'N/A' else np.nan,
                                'Tiempo': params_act,
                                'Distribucion': distribucion_act,
                                'Frentes_Aplicables': frentes_seleccionados_act,
                                'Es_Personalizada': True
                            }

                            found = False
                            for i, act in enumerate(st.session_state.actividades_adicionales):
                                if act['Actividad'] == nombre_act.strip():
                                    st.session_state.actividades_adicionales[i] = nueva_actividad_data
                                    found = True
                                    break

                            if not found:
                                st.session_state.actividades_adicionales.append(nueva_actividad_data)

                            # El cambio se aplica de inmediato (igual que en el Menu 5): se
                            # fuerza un rerun para que el resto de la app quede sincronizado
                            # sin que el usuario tenga que volver a tocar el Paso 2 a mano.
                            # Sin st.form ya no hay clear_on_submit automatico: se limpia el
                            # nombre a mano para que el campo quede listo para la siguiente
                            # actividad (los demas widgets no necesitan limpiarse: recurso y
                            # distribucion tiene sentido que mantengan el ultimo valor usado).
                            st.session_state["_reset_nueva_act_nombre"] = True
                            st.success(f"Actividad '{nombre_act}' añadida/actualizada correctamente.")
                            reiniciar_resultados_simulacion(f"agregaste/actualizaste la actividad personalizada '{nombre_act}' (Actividades Personalizadas)")
                            st.rerun()
                        else:
                            st.error("Por favor, ingrese el nombre, los parámetros de tiempo y seleccione al menos un túnel aplicable.")

                if st.session_state.actividades_adicionales:
                    st.markdown("#### Actividades Personalizadas Agregadas:")
                    df_adicionales = pd.DataFrame(st.session_state.actividades_adicionales)
                    df_adicionales_display = df_adicionales[['Actividad', 'Recurso', 'Distribucion', 'Tiempo', 'Frentes_Aplicables']].copy()
                    _sec_final = actividades_nivel[actividades_nivel['Es_Personalizada'] == True].set_index('Actividad')['Secuencia'].to_dict()
                    df_adicionales_display.insert(0, 'Secuencia', df_adicionales['Actividad'].map(_sec_final))
                    mostrar_tabla(df_adicionales_display, width='stretch', hide_index=True)

                    col_del1, col_del2 = st.columns([2, 1])
                    with col_del1:
                        nombres_actividades_act = [act['Actividad'] for act in st.session_state.actividades_adicionales]
                        actividad_a_eliminar = st.selectbox(
                            "Seleccionar actividad a eliminar",
                            nombres_actividades_act,
                            key="select_actividad_eliminar"
                        )
                        if st.button("Eliminar actividad seleccionada", key="eliminar_una_adicional", type="primary"):
                            st.session_state.actividades_adicionales = [
                                act for act in st.session_state.actividades_adicionales
                                if act['Actividad'] != actividad_a_eliminar
                            ]
                            st.success(f"Actividad '{actividad_a_eliminar}' eliminada correctamente.")
                            reiniciar_resultados_simulacion(f"eliminaste la actividad personalizada '{actividad_a_eliminar}' (Actividades Personalizadas)")
                            st.rerun()
                    with col_del2:
                        st.markdown("&nbsp;")
                        if st.button("Limpiar Actividades Personalizadas", key="limpiar_adicionales", type="primary"):
                            st.session_state.actividades_adicionales = []
                            reiniciar_resultados_simulacion("limpiaste todas las actividades personalizadas (Actividades Personalizadas)")
                            st.rerun()

        _menu_6_actividades_personalizadas(recursos_nivel, frentes_disponibles, nivel_seleccionado, actividades_nivel)

        # ----------------------------------------------------
        # Configuración de Restricciones Geológicas
        # ----------------------------------------------------
        # Todo el Menu 10 vive dentro de un @st.fragment, igual que los Menus 5, 7, 9 y 16:
        # "Guardar Restricción Geológica" y "Limpiar Todas las Restricciones Geológicas"
        # disparaban un rerun de la app COMPLETA. Encerrado en el fragment, esos botones
        # solo vuelven a ejecutar este bloque.
        @st.fragment
        def _menu_7_restricciones_geologicas(recursos_nivel, frentes_disponibles):

            # ======================================================================
            # #####Menu 10 — Configuración de Restricciones Geológicas#####
            # ======================================================================
            with seccion_menu('10', 'Configuración de Restricciones Geológicas'):

                with st.expander("Añadir/Editar Restricción Geológica"):
                    with st.form("form_res_geo", clear_on_submit=True):
                        st.markdown("#### Definir Restricción")
                    
                        col_g1, col_g2 = st.columns(2)
                        with col_g1:
                            res_geo_nombre = st.text_input("Nombre de la Restricción (único)", key="res_geo_nombre")
                            res_geo_frentes = st.multiselect(
                                "Túneles (Frentes) Afectados",
                                frentes_disponibles,
                                key="res_geo_frentes"
                            )
                        with col_g2:
                            res_geo_avance = st.number_input(
                                "Castigo/Bonus a Metros por Ciclo (%)",
                                help="Ej: -50 para un 50% menos de avance. +10 para un 10% más.",
                                value=0,
                                step=5,
                                key="res_geo_avance"
                            )

                        subtitulo_menu("Ponderadores de Tiempo de Actividad por Recurso (%)",
                                       ayuda="Afecta la duración de las actividades. Ej: +30 para que la actividad dure un 30% más, -20 para que dure un 20% menos.")

                        ponderadores_recursos = {}
                        recursos_list_geo = list(recursos_nivel['Recurso'].unique())
                        num_cols_geo = 3
                        cols_geo = st.columns(num_cols_geo)
                        for i, recurso in enumerate(recursos_list_geo):
                            with cols_geo[i % num_cols_geo]:
                                ponderador = st.number_input(
                                    f"{recurso} (%)",
                                    value=0,
                                    step=5,
                                    key=f"ponderador_{recurso}"
                                )
                                if ponderador != 0:
                                    ponderadores_recursos[recurso] = ponderador

                        submitted_geo = st.form_submit_button("Guardar Restricción Geológica", type="primary")

                        if submitted_geo:
                            if res_geo_nombre and res_geo_frentes:
                                nueva_res_geo = {
                                    'nombre': res_geo_nombre.strip(),
                                    'frentes_aplicables': res_geo_frentes,
                                    'castigo_avance': res_geo_avance,
                                    'ponderadores_recursos': ponderadores_recursos
                                }

                                # Si ya existe una con ese nombre, la actualiza. Si no, la añade.
                                found = False
                                for i, res in enumerate(st.session_state.restricciones_geologicas):
                                    if res['nombre'] == res_geo_nombre.strip():
                                        st.session_state.restricciones_geologicas[i] = nueva_res_geo
                                        found = True
                                        break
                                if not found:
                                    st.session_state.restricciones_geologicas.append(nueva_res_geo)
                            
                                st.success(f"Restricción **'{res_geo_nombre}'** guardada.")
                                reiniciar_resultados_simulacion(f"guardaste la restricción geológica '{res_geo_nombre}' (Restricciones Geológicas)")

                            else:
                                st.error("Por favor, ingrese el nombre y seleccione al menos un túnel afectado.")

                if st.session_state.restricciones_geologicas:
                    st.markdown("#### Restricciones Geológicas Configuradas:")
                
                    display_data = []
                    for res in st.session_state.restricciones_geologicas:
                        ponderadores_str = ", ".join([f"{k}: {v}%" for k, v in res['ponderadores_recursos'].items()])
                        if not ponderadores_str:
                            ponderadores_str = "Ninguno"
                    
                        display_data.append({
                            "Nombre": res['nombre'],
                            "Túneles Afectados": ", ".join(res['frentes_aplicables']),
                            "Ajuste Avance": f"{res['castigo_avance']}%",
                            "Ponderadores Recursos": ponderadores_str
                        })
                
                    mostrar_tabla(pd.DataFrame(display_data), width='stretch', hide_index=True)

                    if st.button("Limpiar Todas las Restricciones Geológicas", key="limpiar_res_geo", type="primary"):
                        st.session_state.restricciones_geologicas = []
                        reiniciar_resultados_simulacion("limpiaste todas las restricciones geológicas (Restricciones Geológicas)")
                        st.rerun()

                _mwd_ui_menu10(frentes_disponibles, actividades_nivel)

        _menu_7_restricciones_geologicas(recursos_nivel, frentes_disponibles)

        # ----------------------------------------------------
        # Configuración de Fallas de Equipos
        # ----------------------------------------------------
        # Todo el Menu 11 vive dentro de un @st.fragment, igual que los Menus 5, 7, 9 y 10:
        # "Guardar Escenario de Falla" y "Limpiar Todos los Escenarios de Falla" disparaban
        # un rerun de la app COMPLETA. Encerrado en el fragment, esos botones solo vuelven
        # a ejecutar este bloque.
        @st.fragment
        def _menu_8_fallas_equipos(recursos_nivel, frentes_disponibles):

            # ======================================================================
            # #####Menu 11 — Configuración de Fallas de Equipos#####
            # ======================================================================
            with seccion_menu('11', 'Configuración de Fallas de Equipos'):

                with st.expander("Añadir/Editar Escenario de Falla"):
                    # Nota: ya NO se usa st.form aquí (se usaba antes), porque dentro de un
                    # form los widgets no refrescan la vista previa de la distribución hasta
                    # enviar el formulario. Con widgets sueltos + un boton normal, la vista
                    # previa/filtro se actualiza al vuelo, igual que en el Menú 8 y el 12.
                    st.markdown("#### Definir Escenario de Falla")

                    col_f1, col_f2 = st.columns(2)
                    with col_f1:
                        falla_nombre = st.text_input("Nombre del Escenario de Falla (único)", key="falla_nombre")
                    with col_f2:
                        falla_frentes = st.multiselect(
                            "Túneles (Frentes) Afectados",
                            frentes_disponibles,
                            key="falla_frentes"
                        )

                    subtitulo_menu("Probabilidad de Falla por Ciclo y Demora Asociada",
                                   ayuda="Para cada recurso defina la probabilidad (%) de que falle en un ciclo y la demora que produce: fija (Manual) o con una distribución estadística.")

                    fallas_por_recurso = {}
                    recursos_list_falla = list(recursos_nivel['Recurso'].unique())

                    for recurso in recursos_list_falla:
                        with st.expander(f"{recurso}", expanded=False):
                            prob = st.number_input(
                                f"Probabilidad de Falla (%) para {recurso}",
                                min_value=0, max_value=100, value=0, step=1,
                                key=f"falla_prob_{recurso}"
                            )

                            modo_demora_falla = st.radio(
                                f"Demora por Falla — {recurso}",
                                ["Manual (horas fijas)", "Distribución estadística"],
                                horizontal=True, key=f"falla_modo_demora_{recurso}"
                            )

                            demora_manual_falla = None
                            distribucion_falla = None
                            params_dist_falla = None

                            if modo_demora_falla == "Manual (horas fijas)":
                                demora_manual_falla = st.number_input(
                                    f"Horas de Demora por Falla para {recurso}",
                                    min_value=0.0, value=0.0, step=0.5,
                                    key=f"falla_demora_{recurso}"
                                )
                            else:
                                # Mismo patrón que Menú 8 / Menú 12: incluye 'Cte' y todas las
                                # distribuciones de DISTRIBUTION_PARAM_INFO, con vista previa
                                # gráfica y filtro de área bajo la curva.
                                distribucion_falla = st.selectbox(
                                    f"Distribución de la Demora por Falla — {recurso} (horas)",
                                    list(DISTRIBUTION_PARAM_INFO.keys()),
                                    key=f"falla_dist_{recurso}"
                                )
                                info_dist_f = DISTRIBUTION_PARAM_INFO[distribucion_falla]
                                valores_dist_f = {}
                                cols_pf = st.columns(min(3, len(info_dist_f['parametros'])) or 1)
                                for i_p, p in enumerate(info_dist_f['parametros']):
                                    with cols_pf[i_p % len(cols_pf)]:
                                        valores_dist_f[p['key']] = st.number_input(
                                            p['label'], min_value=p.get('min'), value=p['default'],
                                            step=0.01 if p.get('min') == 0.0001 else 0.1,
                                            key=f"falla_param_{recurso}_{distribucion_falla}_{p['key']}"
                                        )
                                params_dist_falla = valores_dist_f['valor'] if distribucion_falla == 'Cte' else valores_dist_f

                                st.markdown("##### Vista Previa de la Demora por Falla:")
                                if distribucion_falla == 'Cte':
                                    st.info(f"Distribución constante: la demora siempre es **{params_dist_falla:.3f} h**.")
                                else:
                                    rango_falla = vista_previa_pdf_con_filtro(
                                        distribucion_falla, params_dist_falla,
                                        key_sufijo=f"falla_{recurso}", unidad="h",
                                        titulo_eje_x="Demora por falla (h)"
                                    )
                                    # El rango elegido en la vista previa se guarda como truncamiento
                                    # REAL ('_trunc_lo'/'_trunc_hi'): la simulacion lo respeta.
                                    if rango_falla is not None and isinstance(params_dist_falla, dict):
                                        params_dist_falla['_trunc_lo'] = float(rango_falla[0])
                                        params_dist_falla['_trunc_hi'] = float(rango_falla[1])

                            if prob > 0:
                                fallas_por_recurso[recurso] = {
                                    'probabilidad': prob,
                                    'modo_demora': 'Distribución estadística' if modo_demora_falla != "Manual (horas fijas)" else 'Manual',
                                    'demora': float(demora_manual_falla) if demora_manual_falla is not None else None,
                                    'distribucion': distribucion_falla,
                                    'params_distribucion': params_dist_falla,
                                }

                    if st.button("Guardar Escenario de Falla", type="primary", key="guardar_falla_btn"):
                        if falla_nombre and falla_frentes:
                            nuevo_escenario_falla = {
                                'nombre': falla_nombre.strip(),
                                'frentes_aplicables': falla_frentes,
                                'fallas_por_recurso': fallas_por_recurso
                            }

                            found = False
                            for i, falla in enumerate(st.session_state.fallas_equipos):
                                if falla['nombre'] == falla_nombre.strip():
                                    st.session_state.fallas_equipos[i] = nuevo_escenario_falla
                                    found = True
                                    break
                            if not found:
                                st.session_state.fallas_equipos.append(nuevo_escenario_falla)

                            st.success(f"Escenario de Falla **'{falla_nombre}'** guardado.")
                            reiniciar_resultados_simulacion(f"guardaste el escenario de falla '{falla_nombre}' (Fallas de Equipos)")
                            st.rerun()
                        else:
                            st.error("Por favor, ingrese el nombre y seleccione al menos un túnel afectado.")

                if st.session_state.fallas_equipos:
                    st.markdown("#### Escenarios de Falla Configurados:")

                    display_data_falla = []
                    for falla in st.session_state.fallas_equipos:
                        partes_detalle = []
                        for k, v in falla['fallas_por_recurso'].items():
                            if v.get('modo_demora') == 'Distribución estadística':
                                demora_txt = f"distribución '{v.get('distribucion')}'"
                            else:
                                demora_txt = f"{v.get('demora', 0):g}h fija"
                            partes_detalle.append(f"{k} ({v['probabilidad']}% prob, {demora_txt})")
                        detalles_falla = ", ".join(partes_detalle) if partes_detalle else "Ninguno"

                        display_data_falla.append({
                            "Nombre Escenario": falla['nombre'],
                            "Túneles Afectados": ", ".join(falla['frentes_aplicables']),
                            "Detalles de Falla": detalles_falla
                        })

                    mostrar_tabla(pd.DataFrame(display_data_falla), width='stretch', hide_index=True)

                    if st.button("Limpiar Todos los Escenarios de Falla", key="limpiar_fallas", type="primary"):
                        st.session_state.fallas_equipos = []
                        reiniciar_resultados_simulacion("limpiaste todos los escenarios de falla (Fallas de Equipos)")
                        st.rerun()

        _menu_8_fallas_equipos(recursos_nivel, frentes_disponibles)

        # ----------------------------------------------------
        # Mantenciones Programadas de Equipos
        # ----------------------------------------------------

        # ======================================================================
        # #####Menu 12 — Mantenciones Programadas de Equipos#####
        # ======================================================================
        def _bloque_config_mantencion(key_prefix):
            """Renderiza los campos de UNA configuración de mantención (inicio, duración
            manual o por distribución, probabilidad de éxito y, si corresponde, una
            distribución PROPIA para el tiempo de cada reintento tras una falla) y
            devuelve un dict con todo lo ingresado. Se reutiliza tanto para "misma
            configuración para todas las unidades" como para cada unidad configurada
            por separado (variando solo key_prefix, para que los widgets no choquen)."""
            col_t1, col_t2, col_t3 = st.columns(3)
            with col_t1:
                modo_tiempo = st.radio(
                    "Unidad de tiempo para inicio/fin", ["Turnos", "Horas"],
                    horizontal=True, key=f"{key_prefix}_modo_tiempo"
                )
            with col_t2:
                inicio = st.number_input(
                    f"Inicio ({'turno' if modo_tiempo == 'Turnos' else 'hora'})",
                    min_value=1.0 if modo_tiempo == 'Turnos' else 0.0,
                    value=1.0, step=1.0, key=f"{key_prefix}_inicio"
                )
            with col_t3:
                if modo_tiempo == 'Horas':
                    modo_duracion_raw = st.radio(
                        "Duración de la mantención",
                        ["Manual (indicar fin)", "Distribución estadística"],
                        key=f"{key_prefix}_modo_duracion"
                    )
                else:
                    # Turnos = bloques ENTEROS de turnos: no hay distribución de duración.
                    modo_duracion_raw = "Manual (indicar fin)"
                    ayuda_inline("En modo Turnos la mantención ocupa turnos completos (sin distribución). Para usar una distribución de duración, elija 'Horas'.", "Duración en modo Turnos")

            fin = None
            distribucion = None
            params_dist = None

            if modo_duracion_raw == "Manual (indicar fin)":
                fin = st.number_input(
                    f"Fin ({'turno, inclusive' if modo_tiempo == 'Turnos' else 'hora'})",
                    min_value=inicio, value=(inicio if modo_tiempo == 'Turnos' else inicio + 5.0),
                    step=1.0, key=f"{key_prefix}_fin"
                )
                if modo_tiempo == 'Turnos':
                    n_t = int(round(fin - inicio)) + 1
                    lista_t = ", ".join(str(int(inicio) + k) for k in range(min(n_t, 12))) + (" ..." if n_t > 12 else "")
                    st.caption(
                        f"El equipo estará en mantención durante {n_t} turno(s) completo(s): turno(s) {lista_t} "
                        f"(del turno {inicio:g} al {fin:g}, ambos incluidos)."
                    )
                else:
                    st.caption(
                        f"Duración fija: {fin - inicio:g} hora(s) por mantención "
                        f"(desde la hora {inicio:g} hasta la hora {fin:g})."
                    )
            else:
                # Mismo patrón que el Menú 8 (Modificar Distribuciones de Actividades):
                # incluye 'Cte' (duración fija pero elegida como distribución) y todas las
                # distribuciones de DISTRIBUTION_PARAM_INFO, con vista previa gráfica y
                # filtro de área bajo la curva ("mayor a", "menor a", "entre").
                distribucion = st.selectbox(
                    "Distribución de la Duración de Mantención (horas)",
                    list(DISTRIBUTION_PARAM_INFO.keys()),
                    key=f"{key_prefix}_dist"
                )
                info_dist = DISTRIBUTION_PARAM_INFO[distribucion]
                valores_dist = {}
                cols_params = st.columns(min(3, len(info_dist['parametros'])) or 1)
                for i_p, p in enumerate(info_dist['parametros']):
                    with cols_params[i_p % len(cols_params)]:
                        valores_dist[p['key']] = st.number_input(
                            p['label'], min_value=p.get('min'), value=p['default'],
                            step=0.01 if p.get('min') == 0.0001 else 0.1,
                            key=f"{key_prefix}_param_{distribucion}_{p['key']}"
                        )
                params_dist = valores_dist['valor'] if distribucion == 'Cte' else valores_dist

                st.markdown("##### Vista Previa de la Distribución de Duración:")
                if distribucion == 'Cte':
                    st.info(f"Distribución constante: la mantención siempre dura **{params_dist:.3f} h**.")
                else:
                    rango_mant = vista_previa_pdf_con_filtro(
                        distribucion, params_dist,
                        key_sufijo=f"{key_prefix}_preview", unidad="h",
                        titulo_eje_x="Duración de la mantención (h)"
                    )
                    if rango_mant is not None and isinstance(params_dist, dict):
                        params_dist['_trunc_lo'] = float(rango_mant[0])
                        params_dist['_trunc_hi'] = float(rango_mant[1])

            prob_exito = st.number_input(
                "Probabilidad de Éxito de la Mantención (%)",
                min_value=1, max_value=100, value=100, step=1,
                key=f"{key_prefix}_prob_exito",
                help="Si la mantención fracasa, el equipo debe volver a mantención antes de poder "
                     "trabajar de nuevo (se reintenta) hasta que una intentona tenga éxito o se "
                     "acabe el horizonte simulado."
            )

            modo_reintento = 'Igual a la mantención original'
            distribucion_reintento = None
            params_dist_reintento = None

            if prob_exito < 100 and modo_tiempo == 'Turnos':
                ayuda_inline("Si la mantención falla se repite el mismo bloque de turnos inmediatamente después (en modo Turnos no hay distribución de reintento).", f"Reintentos con {prob_exito}% de éxito")
            if prob_exito < 100 and modo_tiempo == 'Horas':
                st.caption(f"Intentos promedio hasta completar la mantención: ~{100 / prob_exito:.1f}")
                st.markdown("**Duración del reintento tras una falla**",
                            help="Tiempo que tarda el equipo en poder volver a funcionar antes del siguiente reintento.")
                modo_reintento = st.radio(
                    "Duración de cada reintento tras una falla",
                    ["Igual a la mantención original", "Definir una distribución propia para el reintento"],
                    key=f"{key_prefix}_modo_reintento",
                    help="'Igual a la mantención original' repite la misma duración/distribución "
                         "de arriba en cada reintento. La otra opción permite modelar, por ejemplo, "
                         "un tiempo de reintento distinto (y con su propia incertidumbre) al de la "
                         "mantención planificada inicialmente."
                )
                if modo_reintento == "Definir una distribución propia para el reintento":
                    distribucion_reintento = st.selectbox(
                        "Distribución del Tiempo de Reintento (horas)",
                        list(DISTRIBUTION_PARAM_INFO.keys()),
                        key=f"{key_prefix}_dist_reintento"
                    )
                    info_dist_r = DISTRIBUTION_PARAM_INFO[distribucion_reintento]
                    valores_dist_r = {}
                    cols_params_r = st.columns(min(3, len(info_dist_r['parametros'])) or 1)
                    for i_p, p in enumerate(info_dist_r['parametros']):
                        with cols_params_r[i_p % len(cols_params_r)]:
                            valores_dist_r[p['key']] = st.number_input(
                                p['label'], min_value=p.get('min'), value=p['default'],
                                step=0.01 if p.get('min') == 0.0001 else 0.1,
                                key=f"{key_prefix}_param_reintento_{distribucion_reintento}_{p['key']}"
                            )
                    params_dist_reintento = valores_dist_r['valor'] if distribucion_reintento == 'Cte' else valores_dist_r

                    st.markdown("##### Vista Previa del Tiempo de Reintento:")
                    if distribucion_reintento == 'Cte':
                        st.info(f"Reintento de duración constante: **{params_dist_reintento:.3f} h**.")
                    else:
                        rango_reint = vista_previa_pdf_con_filtro(
                            distribucion_reintento, params_dist_reintento,
                            key_sufijo=f"{key_prefix}_preview_reintento", unidad="h",
                            titulo_eje_x="Tiempo de reintento (h)"
                        )
                        if rango_reint is not None and isinstance(params_dist_reintento, dict):
                            params_dist_reintento['_trunc_lo'] = float(rango_reint[0])
                            params_dist_reintento['_trunc_hi'] = float(rango_reint[1])

            return {
                'modo_tiempo': modo_tiempo,
                'inicio': float(inicio),
                'modo_duracion': 'Distribución estadística' if modo_duracion_raw != "Manual (indicar fin)" else 'Manual (inicio-fin)',
                'fin': float(fin) if fin is not None else None,
                'distribucion': distribucion,
                'params_distribucion': params_dist,
                'probabilidad_exito': float(prob_exito),
                'modo_reintento': modo_reintento,
                'distribucion_reintento': distribucion_reintento,
                'params_distribucion_reintento': params_dist_reintento,
            }

        def _menu_8b_mantenciones_programadas(recursos_nivel):
            with seccion_menu('12', 'Mantenciones Programadas de Equipos', ayuda=(
                    "Planifique cuándo un EQUIPO ESPECÍFICO (no un túnel) sale de servicio por mantención. Mientras dura, "
                    "esa unidad no está disponible y el frente espera; si hay otras unidades del mismo tipo, el motor usa "
                    "la que esté libre. Es independiente de la falla aleatoria del menú de Fallas.")):

                recursos_list_mant = sorted(recursos_nivel['Recurso'].unique())
                if not recursos_list_mant:
                    st.info("No hay recursos configurados en este nivel todavía.")
                    return

                duracion_turno_mant = float(st.session_state.get('duracion_turno_base', 8.0))

                with st.expander("Añadir/Editar Mantención Programada"):
                    st.markdown("#### Definir Mantención Programada")

                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        mant_nombre = st.text_input("Nombre base de la Mantención (único por unidad)", key="mant_nombre")
                    with col_m2:
                        mant_recurso = st.selectbox("Equipo (tipo de recurso)", recursos_list_mant, key="mant_recurso")

                    cantidad_configurada = int(math.ceil(float(
                        st.session_state.get('recursos_config', {}).get(mant_recurso, {}).get('cantidad', 1) or 1
                    )))
                    opciones_instancias = list(range(1, cantidad_configurada + 1))
                    # Si el usuario cambia de 'Equipo' y la seleccion anterior de unidades ya no
                    # es valida para el nuevo recurso (ej. tenia elegidas la 3 y 4, y el nuevo
                    # recurso solo tiene 2 unidades), se sanea ANTES de renderizar el multiselect
                    # para evitar un error de Streamlit por opciones fuera de rango.
                    if "mant_instancias_sel" in st.session_state:
                        st.session_state["mant_instancias_sel"] = [
                            n for n in st.session_state["mant_instancias_sel"] if n in opciones_instancias
                        ]
                    instancias_seleccionadas = st.multiselect(
                        f"Unidad(es) de '{mant_recurso}' a llevar a mantención "
                        f"(hay {cantidad_configurada} configurada(s))",
                        opciones_instancias, default=opciones_instancias[:1],
                        key="mant_instancias_sel",
                        help="Puedes elegir varias unidades a la vez: por ejemplo, si hay 4 Scoop, "
                             "puedes mandar solo la N°1 y la N°3 a mantención."
                    )

                    if not instancias_seleccionadas:
                        st.warning("Selecciona al menos una unidad para configurar su mantención.")
                    else:
                        modo_multi = "Misma configuración para todas"
                        if len(instancias_seleccionadas) > 1:
                            modo_multi = st.radio(
                                "¿Cómo configurar las unidades seleccionadas?",
                                ["Misma configuración para todas", "Configurar cada unidad por separado"],
                                horizontal=True, key="mant_modo_multi",
                                help="'Misma configuración' aplica el mismo turno/hora de inicio, misma "
                                     "duración y misma probabilidad de éxito a todas las unidades elegidas "
                                     "(cada una sortea igual sus propios resultados al azar en cada "
                                     "simulación). 'Por separado' permite, por ejemplo, mandar la Unidad 1 "
                                     "a mantención en el turno 3 y la Unidad 2 en el turno 10, cada una con "
                                     "su propia duración y % de éxito."
                            )

                        configs_por_instancia = {}
                        if modo_multi == "Misma configuración para todas":
                            etiquetas_sel = ", ".join(f"#{n}" for n in instancias_seleccionadas)
                            st.markdown(f"##### Configuración común para: {mant_recurso} {etiquetas_sel}")
                            cfg_comun = _bloque_config_mantencion("mant_comun")
                            for n in instancias_seleccionadas:
                                configs_por_instancia[n] = cfg_comun
                        else:
                            for n in instancias_seleccionadas:
                                with st.container(border=True):
                                    st.markdown(f"##### Configuración para: {mant_recurso} #{n}")
                                    configs_por_instancia[n] = _bloque_config_mantencion(f"mant_ind_{mant_recurso}_{n}")

                        if st.button("Guardar Mantención Programada", type="primary", key="guardar_mant_btn"):
                            if not mant_nombre or not mant_nombre.strip():
                                st.error("Por favor, ingrese un nombre para la mantención.")
                            else:
                                nombre_base = mant_nombre.strip()
                                multiples = len(instancias_seleccionadas) > 1
                                nuevas = []
                                for n in instancias_seleccionadas:
                                    cfg = configs_por_instancia[n]
                                    nombre_final = f"{nombre_base} — {mant_recurso} #{n}" if multiples else nombre_base
                                    nuevas.append({
                                        'nombre': nombre_final,
                                        'recurso': mant_recurso,
                                        'instancia_id': n - 1,  # 0-based, igual que 'id' de instancias_recursos
                                        **cfg
                                    })
                                for nueva_mantencion in nuevas:
                                    found = False
                                    for i, m_exist in enumerate(st.session_state.mantenciones_programadas):
                                        if m_exist['nombre'] == nueva_mantencion['nombre']:
                                            st.session_state.mantenciones_programadas[i] = nueva_mantencion
                                            found = True
                                            break
                                    if not found:
                                        st.session_state.mantenciones_programadas.append(nueva_mantencion)
                                st.success(
                                    f"Se guardó la mantención de **'{nombre_base}'** para "
                                    f"{len(nuevas)} unidad(es) de '{mant_recurso}'."
                                )
                                reiniciar_resultados_simulacion(f"guardaste la mantención programada '{nombre_base}' (Mantenciones Programadas)")
                                st.rerun()

                if st.session_state.mantenciones_programadas:
                    st.markdown("#### Mantenciones Programadas Configuradas:")
                    display_data_mant = []
                    for m in st.session_state.mantenciones_programadas:
                        unidad_txt = 'turno(s)' if m['modo_tiempo'] == 'Turnos' else 'hora(s)'
                        if m['modo_tiempo'] == 'Turnos':
                            duracion_txt = f"turnos {m['inicio']:g} a {m['fin']:g} (completos, ambos incluidos)"
                        elif m['modo_duracion'] == 'Manual (inicio-fin)':
                            duracion_txt = f"{m['inicio']:g} a {m['fin']:g} ({unidad_txt}, manual)"
                        else:
                            duracion_txt = f"desde {m['inicio']:g} {unidad_txt}, duración '{m['distribucion']}'"
                        reintento_txt = (
                            "igual a la original" if m.get('modo_reintento', 'Igual a la mantención original') == 'Igual a la mantención original'
                            else f"propia ('{m.get('distribucion_reintento')}')"
                        )
                        display_data_mant.append({
                            "Nombre": m['nombre'],
                            "Equipo": f"{m['recurso']} #{m['instancia_id'] + 1}",
                            "Ventana": duracion_txt,
                            "Prob. Éxito (%)": m['probabilidad_exito'],
                            "Duración Reintento": reintento_txt,
                        })
                    mostrar_tabla(pd.DataFrame(display_data_mant), width='stretch', hide_index=True)

                    n_turnos_mant = int(st.session_state.get('numero_turnos', 0))
                    if n_turnos_mant > 0:
                        with st.expander("Ver calendario de mantenciones por turno", expanded=False):
                            estado_mant = construir_calendario_mantenciones(
                                st.session_state.mantenciones_programadas, duracion_turno_mant, n_turnos_mant
                            )
                            st.plotly_chart(
                                graficar_calendario_mantenciones(estado_mant, n_turnos_mant),
                                width='stretch', key="calendario_mantenciones_chart"
                            )
                            ayuda_inline("Vista previa determinística (usa el fin manual o el promedio de la distribución elegida). La simulación real vuelve a sortear duración y éxito en cada corrida Monte Carlo.", "Cómo leer el calendario")

                    if st.button("Limpiar Todas las Mantenciones Programadas", key="limpiar_mantenciones", type="primary"):
                        st.session_state.mantenciones_programadas = []
                        reiniciar_resultados_simulacion("limpiaste todas las mantenciones programadas (Mantenciones Programadas)")
                        st.rerun()

        _menu_8b_mantenciones_programadas(recursos_nivel)

        # ----------------------------------------------------
        # Selección de Frentes
        # ----------------------------------------------------

        # ======================================================================
        # #####Menu 13 — Selección de Frentes a Simular#####
        # ======================================================================
        with seccion_menu('13', 'Selección de Frentes a Simular'):
            frentes_disponibles_sorted = sorted(frentes_nivel['Frentes'].unique())
            frentes_seleccionados = st.multiselect(
                "Seleccione los frentes a simular",
                frentes_disponibles_sorted,
                default=frentes_disponibles_sorted,
                key="frentes_seleccionados"
            )

        # ----------------------------------------------------
        # Configuración de Restricciones
        # ----------------------------------------------------

        # ======================================================================
        # #####Menu 14 — Configuración de Restricciones Físicas (FH, OC, R)#####
        # ======================================================================
        with seccion_menu('14', 'Configuración de Restricciones Físicas (FH, OC, R)'):

            restricciones_activas = st.toggle(
                "Aplicar restricciones físicas (FH, OC, R)", value=True, key="restricciones_fisicas_activas",
                help="Activado: las restricciones FH, ObrasCiviles y Restriccion generan demora en la simulación. "
                     "Desactivado: se ignoran por completo (sin demoras por restricciones) y se ocultan los parámetros de esta sección.")
            if restricciones_activas:
                col_rest_1, col_rest_2 = st.columns(2)
                with col_rest_1:
                    radio_restriccion = st.number_input(
                        "Radio de Impacto FH/OC (metros)",
                        min_value=0.1, value=5.0, step=0.5, key="radio_restriccion_input"
                    )
                with col_rest_2:
                    demora_horas = st.number_input(
                        "Demora Aplicada si hay Impacto (horas)",
                        min_value=0.0, value=2.0, step=0.1, key="demora_horas_input"
                    )
                subtitulo_menu("Radios por tipo de restriccion de turno",
                               ayuda="Radio de influencia por tipo, usado para la hoja Restriccion. Valores del motor original: Polvorazo = 80 m, PA FH = 40 m.")
                tipos_restriccion = sorted(set(st.session_state.df_res['Tipo'].dropna().astype(str)) | set(DEFAULT_RESTRICTION_RADII.keys()))
                df_radios = pd.DataFrame({
                    'Tipo': tipos_restriccion,
                    'Radio (m)': [float(DEFAULT_RESTRICTION_RADII.get(tipo, radio_restriccion)) for tipo in tipos_restriccion]
                })
                df_radios_editado = st.data_editor(
                    df_radios,
                    column_config={
                        'Tipo': st.column_config.TextColumn('Tipo', alignment="center"),
                        'Radio (m)': st.column_config.NumberColumn('Radio (m)', alignment="center"),
                    },
                    hide_index=True,
                    width='stretch',
                    key="radios_restriccion_editor"
                )
                radios_por_tipo = {
                    str(row['Tipo']): float(row['Radio (m)'])
                    for _, row in df_radios_editado.iterrows()
                    if pd.notna(row['Tipo']) and pd.notna(row['Radio (m)'])
                }
            else:
                radio_restriccion = 0.0
                demora_horas = 0.0
                radios_por_tipo = {}
            st.session_state.radio_restriccion = radio_restriccion
            st.session_state.demora_horas = demora_horas
            st.session_state.radios_por_tipo = radios_por_tipo

            # ---- Aplicación automática + feedback (Menus 13 y 14) ----
            firma_frentes_rest_actual = (
                bool(restricciones_activas),
                tuple(sorted(frentes_seleccionados)),
                round(float(radio_restriccion), 4),
                round(float(demora_horas), 4),
                tuple(sorted((k, round(v, 4)) for k, v in radios_por_tipo.items())),
            )
            firma_frentes_rest_previa = st.session_state.get('_firma_frentes_restricciones')
            if firma_frentes_rest_previa is not None and firma_frentes_rest_actual != firma_frentes_rest_previa:
                if 'resultados' in st.session_state:
                    reiniciar_resultados_simulacion("cambiaste los frentes a simular o las restricciones físicas (Selección de Frentes y Restricciones Físicas)")
                else:
                    st.success("Frentes a simular / restricciones físicas aplicados automáticamente.")
            st.session_state['_firma_frentes_restricciones'] = firma_frentes_rest_actual

            # ----------------------------------------------------
            # Vista Previa de Restricciones (ANTES de simular)
            # ----------------------------------------------------
            if restricciones_activas:
                subtitulo_menu("Vista Previa: ¿Qué restricciones se activarán?",
                               ayuda="Muestra, sin simular, qué filas de FH, ObrasCiviles y Restriccion impactan algún frente seleccionado (y generarán demora) y cuáles quedan fuera de rango. Se recalcula al cambiar el radio, los frentes o los datos.")
                df_preview_rest, resumen_preview_rest = analizar_restricciones_preview(
                    frentes_nivel, st.session_state.df_fh, st.session_state.df_oc, st.session_state.df_res,
                    radio_restriccion, radios_por_tipo, frentes_seleccionados
                )
                if resumen_preview_rest['total'] == 0:
                    st.info("No hay restricciones (FH, ObrasCiviles o Restriccion) cargadas para este nivel.")
                else:
                    n_turnos_config = int(st.session_state.get('numero_turnos', 0))
                    turnos_activos_preview = resumen_preview_rest.get('turnos_activos', [])
                    if resumen_preview_rest['activas'] == 0:
                        st.warning(f"Ninguna de las {resumen_preview_rest['total']} restricciones cargadas se activa con el radio actual: no habrá demoras por restricciones.",
                                   icon=None)
                    else:
                        rango_turnos_preview = _formatear_turnos_como_rangos(turnos_activos_preview)
                        detalle_turnos = f" Turnos afectados: **{rango_turnos_preview}**." if turnos_activos_preview else ""
                        contexto_n_turnos = f" (de un total configurado de {n_turnos_config} turnos)" if n_turnos_config else ""
                        st.info(
                            f"Se activarán **{resumen_preview_rest['activas']} de "
                            f"{resumen_preview_rest['total']}** restricciones cargadas para los frentes "
                            f"seleccionados{contexto_n_turnos}.{detalle_turnos}"
                        )
                    with st.expander("Ver detalle fila por fila (Tipo, Turno, Ubicación, Radio aplicado, Estado)"):
                        mostrar_tabla(df_preview_rest, hide_index=True, width='stretch')
            else:
                df_preview_rest = pd.DataFrame()
            st.session_state.df_preview_restricciones = df_preview_rest

            # ----------------------------------------------------
            # Bloqueo Total de Túneles (Restricción Dura, opcional)
            # ----------------------------------------------------
            subtitulo_menu("Bloqueo Total de Túneles (opcional)", ayuda=(
                "Por defecto, una restricción (Polvorazo, PA FH, FH, ObrasCiviles) solo demora el inicio del ciclo las horas "
                "configuradas y el frente sigue avanzando. Para que un túnel quede con avance 0 mientras dure la restricción "
                "(comportamiento del motor original, shift_State = False), agréguelo en la tabla y deje 'Activar' en 'Sí'. Un túnel bloqueado no ejecuta "
                "actividades y sus recursos quedan libres para los demás frentes."))

            bloqueo_total_activo = st.toggle(
                "Aplicar bloqueo total de túneles", value=False, key="bloqueo_total_activo",
                help="Activado: se habilita la tabla de bloqueos y los túneles indicados quedan con avance 0 durante sus ventanas. "
                     "Desactivado: no se aplica ningún bloqueo (la tabla se conserva por si lo vuelve a activar).")
            if bloqueo_total_activo:
                frentes_sugeridos_bloqueo = sorted({
                    f.strip() for lista in df_preview_rest.loc[
                        df_preview_rest['Estado'] == 'Se activa', 'Frentes Afectados'
                    ] for f in str(lista).split(',') if f.strip() and f.strip() != '—'
                }) if not df_preview_rest.empty else []

                duracion_turno_ui = float(st.session_state.get('duracion_turno_base', 8.0))
                tiempo_limite_ui = float(st.session_state.get('tiempo_limite', 0.0))
                n_turnos_ui = int(st.session_state.get('numero_turnos', 0))

                # Se puede elegir cualquier frente seleccionado en 'Selección de Frentes a Simular'.
                opciones_frente_bloqueo = frentes_disponibles_sorted

                MODO_BLOQUEO_TODA = 'Toda la planificación'
                MODO_BLOQUEO_RANGO = 'Entre Inicio y Fin (turno u hora)'
                columnas_bloqueo = ['Activar', 'Frente', 'Modo', 'Unidad', 'Inicio', 'Fin', 'Duración']
                MODOS_BLOQUEO_RETIRADOS = ('Desde el inicio (N horas/turnos)', 'En un momento específico')
                st.session_state.setdefault('_bloqueos_editor_version', 0)
                if 'df_bloqueos_tuneles_config' not in st.session_state:
                    st.session_state.df_bloqueos_tuneles_config = pd.DataFrame(columns=columnas_bloqueo)
                # Compatibilidad con tablas ya cargadas: agrega 'Activar' (filas previas quedan en 'Sí' para
                # no cambiar lo que ya estaba configurado), pasa los modos retirados a 'Entre Inicio y Fin' y
                # convierte Activar de bool a 'Sí'/'No'. Solo corre cuando hace falta (no en cada rerun), para
                # no refrescar el editor sin necesidad.
                _cfg_b = st.session_state.df_bloqueos_tuneles_config
                _necesita_migrar = ('Activar' not in _cfg_b.columns) or (not _cfg_b.empty and (
                    _cfg_b['Modo'].isin(MODOS_BLOQUEO_RETIRADOS).any()
                    or (~_cfg_b['Activar'].isin(['Sí', 'No'])).any()
                ))
                if _necesita_migrar:
                    _df_b = _cfg_b.copy()
                    if 'Activar' not in _df_b.columns:
                        _df_b.insert(0, 'Activar', 'Sí')
                    for _i in _df_b.index:
                        _m = _df_b.at[_i, 'Modo']
                        if _m == 'Desde el inicio (N horas/turnos)':
                            _dur = pd.to_numeric(_df_b.at[_i, 'Duración'], errors='coerce')
                            _df_b.at[_i, 'Inicio'] = 1.0
                            _df_b.at[_i, 'Fin'] = (float(_dur) if _df_b.at[_i, 'Unidad'] == 'Turnos' else 1.0 + float(_dur)) if pd.notna(_dur) else 1.0
                            _df_b.at[_i, 'Modo'] = MODO_BLOQUEO_RANGO
                        elif _m == 'En un momento específico':
                            _df_b.at[_i, 'Modo'] = MODO_BLOQUEO_RANGO
                    _df_b['Activar'] = _df_b['Activar'].map(
                        lambda v: 'Sí' if (v == 'Sí' or (isinstance(v, (bool, np.bool_)) and bool(v))) else 'No'
                    )
                    st.session_state.df_bloqueos_tuneles_config = _df_b.reindex(columns=columnas_bloqueo)
                    st.session_state['_bloqueos_editor_version'] += 1  # el editor toma la tabla migrada

                # ---- Autocompletar desde la hoja Restriccion: esta YA trae el turno exacto en que
                # cada restriccion se activa (columna 'Turno'), asi que en vez de tipear turno por
                # turno a mano, se arma una fila por cada rango contiguo de turnos activos por frente
                # (quedan editables/eliminables despues en la tabla, nada se aplica a ciegas). ----
                col_auto_1, col_auto_2 = st.columns([3, 1])
                with col_auto_1:
                    ayuda_inline("La hoja Restriccion trae el turno exacto en que cada fila se activa (columna Turno). El botón Autocompletar usa esos turnos para proponer bloqueos totales; quedan como filas editables o eliminables antes de simular.", "Autocompletar desde la hoja Restriccion")
                with col_auto_2:
                    auto_disabled = not frentes_sugeridos_bloqueo
                    if st.button("Autocompletar", width='stretch', disabled=auto_disabled,
                                 help="Agrega una fila de bloqueo total por cada rango de turnos en que la "
                                      "hoja Restriccion ya viene activa, para cada frente afectado. Las filas "
                                      "quedan en 'No': cambie 'Activar' a 'Sí' en las que quiera aplicar."):
                        turnos_por_frente_sugerido = {}
                        if not df_preview_rest.empty:
                            df_r_activa = df_preview_rest[
                                df_preview_rest['Fuente'].astype(str).str.startswith('Restricción (')
                                & (df_preview_rest['Estado'] == 'Se activa')
                            ]
                            for _, fila in df_r_activa.iterrows():
                                turno_val = fila.get('Turno')
                                if pd.isna(turno_val):
                                    continue
                                afectados = [
                                    f.strip() for f in str(fila.get('Frentes Afectados', '')).split(',')
                                    if f.strip() and f.strip() != '—'
                                ]
                                for frente in afectados:
                                    turnos_por_frente_sugerido.setdefault(frente, []).append(int(turno_val))

                        filas_nuevas = [
                            {
                                'Activar': 'No', 'Frente': frente, 'Modo': MODO_BLOQUEO_RANGO, 'Unidad': 'Turnos',
                                'Inicio': turno_ini, 'Fin': turno_fin, 'Duración': (turno_fin - turno_ini + 1)
                            }
                            for frente, turnos in turnos_por_frente_sugerido.items()
                            for turno_ini, turno_fin in _turnos_a_rangos_tuplas(turnos)
                        ]
                        if filas_nuevas:
                            st.session_state.df_bloqueos_tuneles_config = pd.concat(
                                [st.session_state.df_bloqueos_tuneles_config, pd.DataFrame(filas_nuevas)],
                                ignore_index=True
                            )
                            st.session_state['_bloqueos_editor_version'] += 1  # fuerza refresco del data_editor
                            st.success(
                                f"Se agregaron {len(filas_nuevas)} fila(s) de bloqueo total según los turnos "
                                "en que la hoja Restricción ya viene activa. Cambie 'Activar' a 'Sí' en las que quiera aplicar antes de simular."
                            )
                            st.rerun()
                        else:
                            st.info("No hay turnos activos de la hoja Restricción para los frentes seleccionados.")

                # La entrada del editor debe ser ESTABLE entre reruns: si cambia (por ejemplo, al devolverle
                # la tabla ya editada), Streamlit lo trata como un widget nuevo, lo reinicia y se pierde el
                # foco/selección (había que repetir el clic). Por eso el editor recibe una copia 'base' que
                # solo se renueva cuando cambia '_bloqueos_editor_version' (Autocompletar, sincronización
                # Inicio/Fin/Duración o migración); las ediciones del usuario las guarda el propio widget.
                if ('_bloqueos_editor_base' not in st.session_state
                        or st.session_state.get('_bloqueos_base_version') != st.session_state['_bloqueos_editor_version']):
                    st.session_state['_bloqueos_editor_base'] = st.session_state.df_bloqueos_tuneles_config.copy()
                    st.session_state['_bloqueos_base_version'] = st.session_state['_bloqueos_editor_version']
                df_bloqueos_editado = st.data_editor(
                    st.session_state['_bloqueos_editor_base'],
                    num_rows="dynamic",
                    column_config={
                        'Activar': st.column_config.SelectboxColumn(
                            'Activar', options=['Sí', 'No'], default='No', required=True,
                            help="El bloqueo de esa fila solo se aplica cuando está en 'Sí'."
                        ),
                        'Frente': st.column_config.SelectboxColumn(
                            'Frente', options=opciones_frente_bloqueo, required=True
                        ),
                        'Modo': st.column_config.SelectboxColumn(
                            'Modo', required=True,
                            options=[MODO_BLOQUEO_TODA, MODO_BLOQUEO_RANGO],
                            default=MODO_BLOQUEO_TODA
                        ),
                        'Unidad': st.column_config.SelectboxColumn(
                            'Unidad', options=['Horas', 'Turnos'], required=True, default='Turnos'
                        ),
                        'Inicio': st.column_config.NumberColumn(
                            'Inicio (turno u hora)', alignment="center", min_value=1.0, step=1.0, default=1.0,
                            help="Aplica con Modo = 'Entre Inicio y Fin'. "
                                 "Con Unidad='Turnos', turno 1 = tiempo 0 (igual que las restricciones "
                                 "por turno de arriba). NO aplica con 'Toda la planificación'."
                        ),
                        'Fin': st.column_config.NumberColumn(
                            'Fin (turno u hora, inclusive)', alignment="center", min_value=1.0, step=1.0, default=1.0,
                            help="Turno u hora en que TERMINA el bloqueo (inclusive). Aplica con 'Entre Inicio "
                                 "y Fin'. Úsalo para decir literalmente 'del "
                                 "turno 18 al 22'. Si lo cambias, la Duración se recalcula sola "
                                 "(Fin - Inicio + 1 en turnos)."
                        ),
                        'Duración': st.column_config.NumberColumn(
                            'Duración (turnos u horas)', alignment="center", min_value=0.0, step=1.0, default=0.0,
                            help="Cuántos turnos/horas dura el bloqueo. En 'Entre Inicio y Fin' va ligada a Inicio y "
                                 "Fin: si cambias una, la otra se recalcula (turnos: Fin - Inicio + 1, "
                                 "ej. del 1 al 56 son 56 turnos). No aplica con 'Toda la planificación'."
                        ),
                    },
                    hide_index=True, width='stretch',
                    key=f"bloqueos_tuneles_editor_{st.session_state['_bloqueos_editor_version']}"
                )
                # Inicio / Fin / Duración deben decir lo mismo: si el usuario cambia uno, se recalcula el
                # otro (ej. Inicio 1 y Fin 56 -> Duración 56 turnos, Fin inclusivo) y se refresca el editor.
                df_bloqueos_sync, hubo_cambio_sync = sincronizar_inicio_fin_duracion_bloqueos(
                    st.session_state.df_bloqueos_tuneles_config, df_bloqueos_editado
                )
                if hubo_cambio_sync:
                    st.session_state.df_bloqueos_tuneles_config = df_bloqueos_sync
                    st.session_state['_bloqueos_editor_version'] += 1  # fuerza refresco del data_editor
                    st.rerun()
                st.session_state.df_bloqueos_tuneles_config = df_bloqueos_editado

                bloqueos_config = [
                    {
                        'Frente': row['Frente'], 'Modo': row.get('Modo') or MODO_BLOQUEO_TODA,
                        'Unidad': row.get('Unidad') or 'Turnos',
                        'Inicio': row.get('Inicio'), 'Fin': row.get('Fin'), 'Duracion': row.get('Duración')
                    }
                    for _, row in df_bloqueos_editado.iterrows()
                    if pd.notna(row.get('Frente')) and row.get('Activar') == 'Sí'
                ]
                bloqueos_tuneles_dict = calcular_bloqueos_tuneles(
                    bloqueos_config, duracion_turno_ui, tiempo_limite_ui
                ) if tiempo_limite_ui > 0 else {}
                st.session_state.bloqueos_tuneles_config = bloqueos_config
                st.session_state.bloqueos_tuneles_dict = bloqueos_tuneles_dict

                # ---- Estado de las filas: explica por qué una fila puede no cambiar el resumen ni la gráfica ----
                # (fila en 'No' = no se aplica; fila en 'Sí' pero fuera del horizonte simulado = sin efecto)
                def _num_fila(v):
                    try:
                        v = float(v)
                        return str(int(v)) if v == int(v) else str(v)
                    except Exception:
                        return '?'
                _n_si_b, _n_no_b, _sin_efecto_b = 0, 0, []
                for _pos, (_, _fila_b) in enumerate(df_bloqueos_editado.iterrows(), start=1):
                    if pd.isna(_fila_b.get('Frente')):
                        continue
                    if _fila_b.get('Activar') != 'Sí':
                        _n_no_b += 1
                        continue
                    _n_si_b += 1
                    if tiempo_limite_ui > 0 and not calcular_bloqueos_tuneles(
                        [{'Frente': _fila_b['Frente'], 'Modo': _fila_b.get('Modo') or MODO_BLOQUEO_TODA,
                          'Unidad': _fila_b.get('Unidad') or 'Turnos', 'Inicio': _fila_b.get('Inicio'),
                          'Fin': _fila_b.get('Fin'), 'Duracion': _fila_b.get('Duración')}],
                        duracion_turno_ui, tiempo_limite_ui
                    ):
                        _desc = f"fila {_pos} ({_fila_b['Frente']}"
                        if _fila_b.get('Modo') == MODO_BLOQUEO_RANGO:
                            _desc += f": {_num_fila(_fila_b.get('Inicio'))}–{_num_fila(_fila_b.get('Fin'))} {str(_fila_b.get('Unidad') or 'Turnos').lower()}"
                        _sin_efecto_b.append(_desc + ")")
                if _n_si_b or _n_no_b:
                    st.caption(f"Filas en 'Sí' (se aplican): {_n_si_b} · Filas en 'No' (no se aplican): {_n_no_b}")
                if _sin_efecto_b:
                    _horizonte_txt = f"{n_turnos_ui} turnos / {r2(tiempo_limite_ui)} h" if n_turnos_ui > 0 else f"{r2(tiempo_limite_ui)} h"
                    st.warning(
                        f"{len(_sin_efecto_b)} fila(s) en 'Sí' no generan ningún bloqueo porque caen fuera del horizonte "
                        f"simulado ({_horizonte_txt}) o tienen Fin/Duración inválidos: "
                        + ", ".join(_sin_efecto_b[:10]) + ("…" if len(_sin_efecto_b) > 10 else "") + "."
                    )

                if bloqueos_tuneles_dict:
                    partes_resumen_bloqueo = []
                    for frente_bq, intervalos_bq in bloqueos_tuneles_dict.items():
                        rangos_h = ", ".join(f"{r2(ini)}h–{r2(fin)}h" for ini, fin in intervalos_bq)
                        partes_resumen_bloqueo.append(f"**{frente_bq}** ({rangos_h})")
                    st.info(f"{len(bloqueos_tuneles_dict)} túnel(es) con bloqueo total: " + "; ".join(partes_resumen_bloqueo) + ".")
                    with st.expander("Ver detalle de ventanas de bloqueo total configuradas"):
                        mostrar_tabla(
                            resumen_bloqueos_tuneles(bloqueos_tuneles_dict, duracion_turno_ui),
                            hide_index=True, width='stretch'
                        )
                else:
                    st.caption("Sin túneles con bloqueo total. Deje 'Activar' en 'Sí' en las filas que quiera aplicar.")

                # ---- Validación visual turno a turno (restricción por turno vs. bloqueo total) ----
                frentes_para_graficar = sorted(set(frentes_sugeridos_bloqueo) | set(bloqueos_tuneles_dict.keys()))
                if frentes_para_graficar and n_turnos_ui > 0:
                    with st.expander("Ver gráfica de validación por turno (restricción vs. bloqueo total)", expanded=bool(bloqueos_tuneles_dict)):
                        estado_turnos = construir_estado_turnos_frente(
                            frentes_para_graficar, n_turnos_ui, duracion_turno_ui, df_preview_rest, bloqueos_tuneles_dict
                        )
                        st.plotly_chart(
                            graficar_estado_turnos_bloqueo(estado_turnos, n_turnos_ui),
                            width='stretch'
                        )

            else:
                frentes_sugeridos_bloqueo = []
                bloqueos_config = []
                bloqueos_tuneles_dict = {}
                st.session_state.bloqueos_tuneles_config = bloqueos_config
                st.session_state.bloqueos_tuneles_dict = bloqueos_tuneles_dict

            # ---- Reinicio automático si cambia la config de bloqueos (mismo patron que Menus 13-14) ----
            firma_bloqueos_actual = tuple(sorted(
                (b['Frente'], b['Modo'], b['Unidad'], round(float(b['Inicio'] or 0), 4),
                 round(float(b['Fin'] or 0), 4), round(float(b['Duracion'] or 0), 4))
                for b in bloqueos_config
            ))
            firma_bloqueos_previa = st.session_state.get('_firma_bloqueos_tuneles')
            if firma_bloqueos_previa is not None and firma_bloqueos_actual != firma_bloqueos_previa:
                if 'resultados' in st.session_state:
                    reiniciar_resultados_simulacion("cambiaste la configuración de Bloqueo Total de Túneles (Bloqueo Total de Túneles)")
            st.session_state['_firma_bloqueos_tuneles'] = firma_bloqueos_actual

        # ----------------------------------------------------
        # Visualización Geometría
        # ----------------------------------------------------

        # ======================================================================
        # #####Menu 15 — Geometría: Túneles y Restricciones Físicas#####
        # ======================================================================
        with seccion_menu('15', 'Geometría: Túneles y Restricciones Físicas'):
            frentes_completos = st.session_state.df_frentes[
                st.session_state.df_frentes['Nivel'] == nivel_seleccionado
            ].copy()
            # El usuario elige ver la geometría en 2D (gráfico original) o en 3D (túneles con su
            # sección real y restricciones como esferas). Ver menu_11_geometria.
            menu_11_geometria(
                frentes_completos, st.session_state.df_fh, st.session_state.df_oc,
                st.session_state.df_res, radio_restriccion, radios_por_tipo
            )

        # ----------------------------------------------------
        # Ruta de Salida y Ejecutar Simulación
        # ----------------------------------------------------
        # Todo el Menu 16 vive dentro de un @st.fragment: asi, tipear en el campo
        # de ruta de salida o presionar "SIMULAR" solo vuelve a ejecutar este
        # fragmento (y muestra el progreso en vivo dentro de el), en vez de
        # disparar un rerun de la app completa que re-renderiza y "apaga"
        # visualmente los Menus 1 a 15 mientras la simulacion corre.
        @st.fragment
        def _menu_12_ejecutar_simulacion(nivel_seleccionado, frentes_nivel, frentes_seleccionados,
                                          radio_restriccion, demora_horas, actividades_nivel):

            # ======================================================================
            # #####Menu 16 — Ejecutar Simulación#####
            # ======================================================================
            with seccion_menu('16', 'Ejecutar Simulación', ayuda=(
                    "Los parámetros de los menús anteriores se aplican solos al cambiarlos. Si modifica algo después de "
                    "presionar SIMULAR, esa simulación queda obsoleta y se reinicia: vuelva a presionar SIMULAR.")):

                ruta_salida = '' if ES_NUBE else st.text_input(
                    "Ruta de salida para los archivos Excel de resultados",
                    value=st.session_state.get('ruta_salida', ''),
                    placeholder=r"Ejemplo: C:\Users\Usuario\Desktop\Resultados",
                    help="Pega la ruta completa donde quieres guardar los Excel. "
                         "Se crearán los 6 archivos formato David Montenegro + un reporte combinado. "
                         "Si no se define, solo estará disponible la descarga directa.",
                    key="ruta_salida_input"
                )
                st.session_state.ruta_salida = ruta_salida.strip()

                if ruta_salida.strip():
                    if os.path.isabs(ruta_salida.strip()):
                        st.success(f"Los resultados se guardarán en: `{ruta_salida.strip()}`")
                    else:
                        st.warning("La ruta no parece absoluta. Usa una ruta completa (ej: C:\\Carpeta\\Subcarpeta).")
                elif ES_NUBE:
                    st.info("Streamlit Cloud: los resultados no se guardan en carpetas. Tras simular, descargue el "
                            "reporte Excel desde «Exportación de Resultados a Excel».")
                else:
                    st.warning("No se definió ruta de salida. Tras simular, podrás descargar el reporte Excel directamente.")

                if st.button("SIMULAR", type="primary", width='stretch'):
                    # ---- Orquestador de tareas (Redis): evita choques entre simulaciones ----
                    redis_client = get_redis_client()
                    id_proceso_sim = str(uuid.uuid4())
                    lock_adquirido = adquirir_lock_simulacion(redis_client, id_proceso_sim)

                    frentes_simular = ordenar_frentes_por_prioridad(
                        frentes_nivel,
                        frentes_seleccionados,
                        st.session_state.df_fh,
                        st.session_state.df_oc,
                        st.session_state.ruta_critica
                    )
                    if not frentes_simular:
                        st.error("Seleccione al menos un frente")
                    elif not lock_adquirido:
                        st.warning("El orquestador (Redis) indica que ya hay una simulación en curso. "
                                   "Espere a que termine antes de lanzar otra para que los procesos no se traben.")
                    else:
                        tiempo_limite_sim = st.session_state.tiempo_limite
                        metros_avance_sim = st.session_state.metros_avance
                        n_simulaciones_sim = int(st.session_state.n_simulaciones_input)
                        log_evento('INFO', f"Simulación iniciada: {n_simulaciones_sim} escenarios, tiempo límite "
                                           f"{tiempo_limite_sim} h, {metros_avance_sim} m por ciclo, frentes: "
                                           f"{', '.join(map(str, frentes_simular))}")
                        _t0_sim = time.time()
                        _logs_refrescar(force=True)

                        # ---- try/finally: garantiza liberar el lock de Redis pase lo que pase ----
                        # Antes, si simular_avance_con_transporte (u otro paso de este bloque)
                        # lanzaba una excepcion, el flujo cortaba ahi y liberar_lock_simulacion
                        # nunca se ejecutaba: el lock quedaba tomado hasta que expirara su TTL
                        # de seguridad (REDIS_LOCK_TTL, 30 min por defecto), bloqueando el
                        # orquestador para cualquier otra simulacion durante ese rato. El
                        # try/finally asegura que el lock SIEMPRE se libere al salir de este
                        # bloque, haya terminado bien o con error.
                        try:
                          with st.spinner('Ejecutando simulación...'):
                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            status_text.text("Calculando demoras por restricciones...")
                            registrar_tarea_historial(redis_client, id_proceso_sim, "inicio", f"{n_simulaciones_sim} escenarios")
                            actualizar_estado_orquestador(redis_client, "en_progreso", 10, "Calculando demoras por restricciones...")
                            frentes_completos_sim = st.session_state.df_frentes[
                                st.session_state.df_frentes['Nivel'] == nivel_seleccionado
                            ].copy()
                            if st.session_state.get('restricciones_fisicas_activas', True):
                                demoras_constantes, demoras_por_turno = calcular_demoras_restricciones(
                                    frentes_completos_sim, st.session_state.df_fh, st.session_state.df_oc, st.session_state.df_res, 
                                    radio_restriccion, demora_horas, st.session_state.radios_por_tipo
                                )
                            else:
                                # Restricciones físicas desactivadas por el usuario: sin demoras FH/OC/R.
                                demoras_constantes = {f: 0 for f in frentes_completos_sim['Frentes'].unique()}
                                demoras_por_turno = []
                    
                            demoras_constantes_sim = { 
                                k: v for k, v in demoras_constantes.items() if k in frentes_simular 
                            }
                            demoras_por_turno_sim = [ 
                                res for res in demoras_por_turno if res['Frente'] in frentes_simular 
                            ]

                            st.session_state.demoras_constantes = demoras_constantes_sim
                            st.session_state.demoras_por_turno = demoras_por_turno_sim

                            # ---- Bloqueo Total de Túneles (Menu 14) ----
                            # Se recalcula aqui (no solo se reusa lo que quedo en session_state al
                            # editar la tabla) porque tiempo_limite_sim/duracion_turno recien se
                            # conocen con certeza en este punto, y para que quede acotado a
                            # frentes_simular (mismo criterio que demoras_constantes_sim de arriba).
                            duracion_turno_sim = st.session_state.get('duracion_turno_base', 8.0)
                            if '12x2' in st.session_state.sistema_turnos:
                                duracion_turno_sim = 12.0
                            elif '8x3' in st.session_state.sistema_turnos:
                                duracion_turno_sim = 8.0
                            else:
                                duracion_turno_sim = 24.0
                            bloqueos_tuneles_dict = calcular_bloqueos_tuneles(
                                st.session_state.get('bloqueos_tuneles_config', []) if st.session_state.get('bloqueo_total_activo', False) else [],
                                duracion_turno_sim, tiempo_limite_sim
                            )
                            bloqueos_tuneles_dict_sim = {
                                k: v for k, v in bloqueos_tuneles_dict.items() if k in frentes_simular
                            }
                            st.session_state.bloqueos_tuneles_dict_simulado = bloqueos_tuneles_dict_sim
                            if bloqueos_tuneles_dict_sim:
                                status_text.text(
                                    f"{len(bloqueos_tuneles_dict_sim)} túnel(es) con bloqueo total configurado..."
                                )

                            progress_bar.progress(10)

                            frentes_info = {}
                            seccion_col_sim = columna_seccion(frentes_nivel)
                            for frente in frentes_simular:
                                fila_frente = frentes_nivel[frentes_nivel['Frentes'] == frente].iloc[0]
                                distancia = calcular_distancia(
                                    fila_frente['Xi'], fila_frente['Yi'], fila_frente['Xf'], fila_frente['Yf']
                                )
                                avance_planificado_val = fila_frente.get(
                                    'Avance Planificado (m/h)', fila_frente.get('Avance Planificado', 0.0)
                                )
                                seccion_frente = str(fila_frente[seccion_col_sim]) if seccion_col_sim else ''
                                # Si el usuario activo 'Metros por Ciclo variable por Sección' en el
                                # Menu 5, se usa lo que el mismo definio ahi para esta seccion (sin
                                # ningun valor calibrado hardcodeado de por medio). Si esta desactivada
                                # (o la seccion no fue editada), metros_por_seccion queda vacio y todos
                                # los frentes caen al 'Metros por Ciclo por Defecto' generico.
                                metros_por_ciclo_frente = st.session_state.metros_por_seccion.get(
                                    seccion_frente, metros_avance_sim
                                )
                                frentes_info[frente] = {
                                    'distancia': distancia,
                                    'xi': fila_frente['Xi'], 'yi': fila_frente['Yi'],
                                    'xf': fila_frente['Xf'], 'yf': fila_frente['Yf'],
                                    'avance_planificado': avance_planificado_val,
                                    'seccion': seccion_frente,
                                    'metros_por_ciclo': metros_por_ciclo_frente,
                                    'nivel': nivel_seleccionado
                                }
                            st.session_state.frentes_info = frentes_info
                            st.caption("Orden de prioridad usado: " + " -> ".join(f"{i}º {f}" for i, f in enumerate(frentes_simular, 1)))

                            status_text.text(f"Simulando 0/{n_simulaciones_sim} escenarios...")
                            actualizar_estado_orquestador(redis_client, "en_progreso", 50, f"Simulando {n_simulaciones_sim} escenarios...")

                            plan_flota_activo = st.session_state.get('plan_recursos_turno', [])
                            _config_mwd_sim = mwd_config_para_simulacion(list(frentes_info.keys()), actividades_nivel)

                            # Rango de la barra reservado para el loop de Monte Carlo: arranca en 10%
                            # (justo despues del calculo de demoras por restricciones) y llega hasta 95%,
                            # dejando el ultimo tramo para el cierre. Ya NO se corre una segunda
                            # simulacion de "linea base": solo se ejecuta la que el usuario configuro.
                            PROGRESO_INICIO_SIM, PROGRESO_FIN_SIM = 10, 95

                            def _actualizar_progreso_simulacion(completadas, total):
                                fraccion = completadas / total
                                progreso_pct = PROGRESO_INICIO_SIM + int(fraccion * (PROGRESO_FIN_SIM - PROGRESO_INICIO_SIM))
                                progress_bar.progress(min(progreso_pct, PROGRESO_FIN_SIM))
                                status_text.text(f"Simulando {completadas}/{total} escenarios...")
                                if completadas == total or completadas % max(1, total // 10) == 0:
                                    log_evento('INFO', f"Simulando {completadas}/{total} escenarios...")
                                    _logs_refrescar()

                            # Llamada a la función de simulación actualizada
                            resultados, estadisticas_recursos, traza_eventos, ventanas_flota, registro_cambios_flota, avance_por_turno_real_sim = simular_avance_con_transporte(
                                actividades_nivel, 
                                frentes_info, 
                                st.session_state.recursos_config, 
                                tiempo_limite_sim, 
                                metros_avance_sim, 
                                n_simulaciones_sim, 
                                st.session_state.velocidades_config,
                                demoras_constantes_sim,
                                demoras_por_turno_sim,
                                st.session_state.sistema_turnos,
                                st.session_state.restricciones_geologicas,
                                st.session_state.fallas_equipos,
                                plan_flota_activo, # <-- Plan de orquestacion de flota por turno
                                progress_callback=_actualizar_progreso_simulacion,
                                bloqueos_tuneles=bloqueos_tuneles_dict_sim,
                                mantenciones_programadas=st.session_state.mantenciones_programadas,
                                demoras_cambio_turno=st.session_state.get('demoras_cambio_turno'),
                                config_mwd=_config_mwd_sim
                            )

                            # NOTA: antes acá se corría una SEGUNDA simulación completa ("línea base",
                            # con la flota fija sin el plan de orquestación por turno) solo para poder
                            # comparar. Se sacó a pedido del usuario: ahora la app corre UNA sola
                            # simulación, exactamente con los datos que el usuario configuró.
                            resultados_base = None

                            st.session_state.resultados = resultados
                            st.session_state.resultados_base_sin_cambios_flota = resultados_base
                            # Avance por turno REAL (misma fuente que "resultados"/Avance Final), para
                            # que la seccion "Avance Acumulado por Turno" deje de reconstruirlo por
                            # conteo de ciclos en la traza (fuente de la inconsistencia reportada:
                            # mostraba avance en turnos donde el Avance Final real habia quedado en 0).
                            st.session_state.avance_por_turno_real = avance_por_turno_real_sim
                            st.session_state.estadisticas_recursos = estadisticas_recursos
                            st.session_state.traza_eventos = traza_eventos
                            # Se construye el DataFrame UNA sola vez aqui, al terminar de simular, en vez
                            # de hacer pd.DataFrame(traza_eventos) cada vez que la UI hace un rerun (esto
                            # era el principal cuello de botella de "carga progresiva": con cientos de
                            # miles de filas, reconstruirlo 5 veces por rerun es lento y evitable).
                            st.session_state.df_traza_eventos_cache = pd.DataFrame(traza_eventos)
                            # run_id unico por corrida: se usa como parte de la clave de cache en las
                            # funciones @st.cache_data que procesan la traza (groupby, percentiles, Gantt),
                            # para que Streamlit sepa que puede reusar resultados si la traza no cambio.
                            st.session_state.run_id = str(uuid.uuid4())
                            if _config_mwd_sim:
                                st.session_state.mwd_log_geologia = pd.DataFrame(_config_mwd_sim.get('_log_geologia', []))
                                st.session_state.mwd_config_usada = {k: v for k, v in _config_mwd_sim.items()
                                                                     if k not in ('catalogo', '_log_geologia')}
                                st.session_state.mwd_catalogo_usado = _config_mwd_sim['catalogo'].copy()
                                log_evento('SUCCESS', f"Geología MWD: {len(st.session_state.mwd_log_geologia):,} bloques de roca sorteados "
                                                      f"en {n_simulaciones_sim} simulaciones")
                            else:
                                for _k_mwd in ('mwd_log_geologia', 'mwd_config_usada', 'mwd_catalogo_usado'):
                                    st.session_state.pop(_k_mwd, None)
                            st.session_state.ventanas_flota = ventanas_flota
                            st.session_state.registro_cambios_flota = registro_cambios_flota
                            st.session_state.nivel_simulado = nivel_seleccionado
                            st.session_state.duracion_turno_simulada = st.session_state.get('duracion_turno_base', 8.0)
                            st.session_state.numero_turnos_simulado = st.session_state.get('numero_turnos', 62)
                            st.session_state.n_simulaciones_ejecutadas = n_simulaciones_sim
                            st.session_state.tiempo_limite_simulado = tiempo_limite_sim
                            st.session_state.df_actividades_simuladas = actividades_nivel.copy()
                            st.session_state.df_frentes_nivel_simuladas = frentes_nivel.copy()
                            progress_bar.progress(100)
                            status_text.success("Simulación completada con éxito!")

                            # ---- Orquestador de tareas (Redis): cierre de la tarea ----
                            actualizar_estado_orquestador(redis_client, "completado", 100, "Simulación completada con éxito")
                            registrar_tarea_historial(redis_client, id_proceso_sim, "completado", f"{n_simulaciones_sim} escenarios")
                            log_evento('SUCCESS', f"Simulación completada en {time.time() - _t0_sim:.1f} s "
                                                  f"({n_simulaciones_sim} escenarios, {len(frentes_simular)} frentes)")

                            # Forzar un unico rerun completo al terminar: asi los Menus 17-22
                            # (fuera de este fragmento) recogen los resultados nuevos de inmediato,
                            # sin que el usuario tenga que interactuar con algo mas para verlos.
                            st.rerun()
                        except Exception as _e_sim:
                            log_evento('ERROR', f"La simulación falló: {type(_e_sim).__name__}: {_e_sim}")
                            _logs_refrescar(force=True)
                            raise
                        finally:
                            # Se ejecuta SIEMPRE (exito, excepcion en cualquier paso de arriba, o el
                            # propio st.rerun(), que internamente corta el script lanzando una
                            # excepcion de control de Streamlit): asi el lock jamas queda pegado
                            # esperando el TTL de seguridad si algo se cae a mitad de camino.
                            liberar_lock_simulacion(redis_client, id_proceso_sim)

        _menu_12_ejecutar_simulacion(nivel_seleccionado, frentes_nivel, frentes_seleccionados,
                                      radio_restriccion, demora_horas, actividades_nivel)

        
        # ----------------------------------------------------
        # Resultados
        # ----------------------------------------------------
        if 'resultados' in st.session_state:
            st.header("Resultados")

            # ======================================================================
            # #####Menu 17 — Resultados de la Simulación#####
            # ======================================================================
            # Qué resultados ver (se define ANTES de las pestañas: aplica a todas)
            with st.container(border=True):
                st.markdown("**¿Qué resultados quieres ver?**")
                # ----------------------------------------------------
                # 13.0 Modo de visualización: Promedio vs Simulación específica
                # ----------------------------------------------------
                # Mostrar solo promedios/percentiles puede ser engañoso (oculta la variabilidad
                # real y mezcla escenarios distintos). Se ofrece la alternativa de ver los
                # resultados de UNA simulación puntual, de forma completamente determinística.
                n_sims_totales_vis = st.session_state.get('n_simulaciones_ejecutadas', 1)
                col_modo_vis, col_sim_vis = st.columns([1.3, 1])
                with col_modo_vis:
                    modo_visualizacion = st.radio(
                        "Modo de visualización de resultados",
                        ["Todas las Simulaciones", "Simulación específica"],
                        horizontal=True,
                        key="modo_visualizacion_resultados"
                    )
                sim_seleccionada_global = None
                if modo_visualizacion == "Simulación específica":
                    with col_sim_vis:
                        sim_seleccionada_global = st.selectbox(
                            "N° de simulación a visualizar",
                            list(range(1, n_sims_totales_vis + 1)),
                            key="sim_seleccionada_global"
                        )
                    st.caption(f"Simulación N° {sim_seleccionada_global} (individual)",
                               help="Resultados de una sola simulación (no promedios ni percentiles). Aplica al Avance Final, Avance por Turno, Utilización de Recursos y Carta Gantt.")

                # Aviso de bloqueos por falta absoluta de recurso: si alguna simulación detectó que
                # un túnel se quedó sin ningún recurso requerido (ahora y para siempre), se muestra
                # aquí de forma explícita antes de cualquier otro resultado.
                df_traza_check = st.session_state.df_traza_eventos_cache
                resumen_bloqueos = calcular_resumen_bloqueos(df_traza_check, st.session_state.run_id)
                if resumen_bloqueos is not None:
                    st.error(
                        "**Uno o más túneles quedaron bloqueados durante la simulación por falta absoluta "
                        "de un recurso requerido** (cantidad 0, sin que el plan de flota lo reincorpore). "
                        "El avance de esos túneles se detuvo en el punto indicado y no continuó."
                    )
                    mostrar_tabla(resumen_bloqueos, width='stretch', hide_index=True)
            
            tab_avance, tab_recursos, tab_gantt, tab_mf, tab_geo, tab_traza, tab_export = st.tabs([
                "Avance", "Recursos y Flota", "Carta Gantt", "Mantenciones y Fallas", "Geología MWD",
                "Traza de Eventos", "Exportar"
            ])
            mwd_mostrar_resultados(tab_geo, sim_seleccionada_global)

            # ======================================================================
            # #####Menu 17 — Resultados: pestaña Avance#####
            # ======================================================================
            with tab_avance, seccion_menu('17', 'Avance de los Túneles'):
                mostrar_resumen_restricciones_aplicadas("Restricciones activas en esta simulación", con_detalle=True)
                mostrar_resumen_demora_cambio_turno()

                # Avance Final
                if sim_seleccionada_global is None:
                    subtitulo_menu("Avance Final Acumulado por Túnel")
                else:
                    subtitulo_menu(f"Avance Final Acumulado por Túnel — Simulación N° {sim_seleccionada_global}")
                df_resultados = dashboard_avance_final(sim_seleccionada_global)

                # ----------------------------------------------------
                # Vista 3D del avance (un color por avance)
                # ----------------------------------------------------
                menu_13_avance_3d(sim_seleccionada_global)

                # ----------------------------------------------------
                # Análisis de Convergencia Monte Carlo (¿alcanzan mis N simulaciones?)
                # ----------------------------------------------------
                with st.expander("Análisis de Convergencia Monte Carlo (¿son suficientes mis N° de simulaciones?)", expanded=False):
                    ayuda_inline("Se analiza el Avance Final por Túnel (una simulación = un valor), la métrica reportada (P10/P50/P90/Esperanza). Sirve para decidir si el N° de simulaciones completas es suficiente.", "Qué se analiza")
                    frente_convergencia = st.selectbox(
                        "Túnel a analizar",
                        list(st.session_state.resultados.keys()),
                        key="frente_convergencia_sel"
                    )
                    valores_convergencia = st.session_state.resultados.get(frente_convergencia, [])

                    if len(valores_convergencia) < 10:
                        st.info("Se necesitan al menos 10 simulaciones para un análisis de convergencia significativo.")
                    else:
                        media_c, error_c, error_rel_c = calcular_error_relativo_convergencia(valores_convergencia)
                        mostrar_tarjetas_metricas([
                            ("Promedio (N total)", f"{media_c:.2f}", "m", True),
                            ("Error estándar", f"{error_c:.3f}", "m", False),
                            ("Error relativo", f"{error_rel_c:.2f}", "%", False),
                        ], ancho_max=230)

                        if error_rel_c < 2:
                            st.success(
                                f"Con N={len(valores_convergencia)} simulaciones, el error relativo del promedio "
                                f"es {error_rel_c:.2f}% (< 2%). El número de simulaciones parece suficiente."
                            )
                        else:
                            st.warning(
                                f"Con N={len(valores_convergencia)} simulaciones, el error relativo del promedio "
                                f"es {error_rel_c:.2f}% (≥ 2%). Podría convenir aumentar el número de simulaciones."
                            )

                        fig_convergencia = graficar_convergencia_montecarlo(
                            valores_convergencia,
                            titulo=f"Convergencia del Avance Final Promedio — {frente_convergencia}"
                        )
                        st.plotly_chart(fig_convergencia, width='stretch')

                        fig_convergencia_pct = graficar_convergencia_percentiles(
                            valores_convergencia,
                            titulo=f"Convergencia de Percentiles del Avance Final — {frente_convergencia}"
                        )
                        st.plotly_chart(fig_convergencia_pct, width='stretch')

                        ayuda_inline("Arriba: cada línea celeste es el promedio acumulado con un orden distinto de las mismas simulaciones; la roja es el promedio final. Abajo: muestra si P10, P50 y P90 se estabilizan al aumentar N (los percentiles extremos necesitan más datos).", "Cómo leer los gráficos")

                st.markdown("---")

                # Avance Acumulado por Turno
                opciones_percentil_turno = {
                    'P0 (Mínimo)': 'p0',
                    'P10': 'p10',
                    'P30': 'p30',
                    'P50 (Mediana)': 'p50',
                    'Esperanza (Promedio)': 'esperanza',
                    'P70': 'p70',
                    'P90': 'p90',
                    'P100 (Máximo)': 'p100',
                }
                if sim_seleccionada_global is None:
                    st.write(f"**Avance Acumulado por Turno**")
                    etiqueta_col_turno = "Avance Prom."
                    percentiles_turno_sel = st.multiselect(
                        "Percentiles a mostrar en la gráfica de avance por turno",
                        list(opciones_percentil_turno.keys()),
                        default=['Esperanza (Promedio)'],
                        key="percentiles_turno_seleccionados",
                        help="Se puede elegir más de un percentil para compararlos en la misma gráfica. "
                             "La tabla siempre muestra la Esperanza (promedio) por túnel/turno."
                    )
                    if not percentiles_turno_sel:
                        percentiles_turno_sel = ['Esperanza (Promedio)']
                else:
                    st.write(f"**Avance Acumulado por Turno — Simulación N° {sim_seleccionada_global}**")
                    etiqueta_col_turno = f"Avance Sim. {sim_seleccionada_global}"
                    percentiles_turno_sel = None
                # Se usa el avance REAL por turno capturado durante la simulacion (misma fuente
                # que "resultados"/Avance Final), en vez de calcular_avance_por_turno (que lo
                # reconstruia contando ciclos en la traza de eventos y podia mostrar avance en
                # turnos donde el Avance Final real habia quedado en 0, por ciclos truncados por
                # tiempo_limite o por bloqueo de recurso). Se mantiene un fallback a la
                # reconstruccion antigua solo por compatibilidad con corridas ya guardadas en
                # session_state antes de este fix, que no tienen 'avance_por_turno_real'.
                if 'avance_por_turno_real' in st.session_state and st.session_state.avance_por_turno_real:
                    avance_turnos = st.session_state.avance_por_turno_real
                else:
                    df_traza_completa = st.session_state.df_traza_eventos_cache
                    avance_turnos = calcular_avance_por_turno(
                        st.session_state.run_id,
                        df_traza_completa, 
                        st.session_state.tiempo_limite, 
                        st.session_state.sistema_turnos, 
                        st.session_state.frentes_info, 
                        st.session_state.metros_avance
                    )

                datos_turno = []
                datos_turno_percentiles = []  # formato largo (Tunel, Turno_Num, Percentil, valor) para la grafica multi-percentil
                if avance_turnos and any(avance_turnos.values()):
                    frentes_keys = list(st.session_state.frentes_info.keys())
                    n_max_turnos = 0
                    for frente in frentes_keys:
                        if frente in avance_turnos:
                            n_max_turnos = max(n_max_turnos, len(avance_turnos[frente]))
                
                    turnos_keys = list(range(1, n_max_turnos + 1))
                
                    for frente in frentes_keys:
                        if frente in avance_turnos:
                            distancia_total_frente = st.session_state.frentes_info.get(frente, {}).get('distancia', None)

                            # Serie de referencia (Esperanza/promedio, o el valor puntual de la
                            # simulación seleccionada): se usa para la tabla y para detectar cuándo
                            # el túnel se estabiliza/termina.
                            lista_valores_serie = []
                            # Series por percentil (solo en modo "Promedio") para la grafica comparativa.
                            series_percentiles = {p: [] for p in (percentiles_turno_sel or [])}

                            for turno in turnos_keys:
                                valores_turno = avance_turnos[frente].get(turno)
                                if not valores_turno:
                                    lista_valores_serie.append(0.0)
                                    for p in series_percentiles:
                                        series_percentiles[p].append(0.0)
                                elif sim_seleccionada_global is None:
                                    arr_turno = np.array(valores_turno, dtype=float)
                                    lista_valores_serie.append(float(np.mean(arr_turno)))
                                    if series_percentiles:
                                        pct_turno = percentiles_reporte(arr_turno)
                                        for p in series_percentiles:
                                            series_percentiles[p].append(float(pct_turno[opciones_percentil_turno[p]]))
                                else:
                                    idx_sim = min(sim_seleccionada_global - 1, len(valores_turno) - 1)
                                    lista_valores_serie.append(float(valores_turno[idx_sim]))
                            valores_serie = np.array(lista_valores_serie)

                            # Umbral de "incremento relevante" entre turnos consecutivos. Se usa
                            # en vez de exigir que el promedio llegue exactamente al 100% de la
                            # meta, porque en el promedio de muchas simulaciones Monte Carlo
                            # siempre puede quedar un pequeño porcentaje de escenarios rezagados
                            # (demoras, fallas de equipos) que nunca alcanzan exactamente la
                            # distancia total, dejando el promedio asintotico sin llegar nunca
                            # al 99.9% exacto.
                            epsilon_avance = max(0.05, (distancia_total_frente or 0.0) * 0.002)

                            # Se busca, de atras hacia adelante, el ultimo turno en el que aun
                            # hubo un incremento relevante de avance. A partir de ahi (si el
                            # tunel sigue avanzando de forma real hasta el ultimo turno, no se
                            # corta nada) se considera que el tunel ya se estabilizo/termino.
                            turno_ultimo_avance_real = len(valores_serie)  # por defecto: no cortar
                            hubo_incremento = False
                            for i in range(len(valores_serie) - 1, 0, -1):
                                if (valores_serie[i] - valores_serie[i - 1]) >= epsilon_avance:
                                    turno_ultimo_avance_real = i + 1  # turnos_keys es 1-based
                                    hubo_incremento = True
                                    break
                            if not hubo_incremento and len(valores_serie) > 0 and valores_serie[0] > 0:
                                # Ya venia avanzando desde el turno 1 y nunca hay un salto grande
                                # detectable (ej. avanza muy poco y parejo): no se corta, se
                                # muestra la serie completa tal cual.
                                turno_ultimo_avance_real = len(valores_serie)

                            fila_frente = {'Tunel': frente}
                            for idx, turno in enumerate(turnos_keys):
                                if turno > turno_ultimo_avance_real:
                                    # El tunel ya se estabilizo (dejo de avanzar de forma
                                    # relevante): no se siguen mostrando resultados.
                                    fila_frente[f'T{turno} {etiqueta_col_turno} (m)'] = np.nan
                                else:
                                    fila_frente[f'T{turno} {etiqueta_col_turno} (m)'] = r2(float(valores_serie[idx]))
                            datos_turno.append(fila_frente)

                            # Formato largo por percentil (solo en modo Promedio), recortado al mismo
                            # punto de estabilizacion que la serie de referencia.
                            if sim_seleccionada_global is None:
                                for p, serie_p in series_percentiles.items():
                                    for idx, turno in enumerate(turnos_keys):
                                        if turno > turno_ultimo_avance_real:
                                            continue
                                        datos_turno_percentiles.append({
                                            'Tunel': frente,
                                            'Turno_Num': turno,
                                            'Percentil': p,
                                            'Avance Acumulado (m)': r2(float(serie_p[idx]))
                                        })

                    if datos_turno:
                        df_turnos = pd.DataFrame(datos_turno)
                        mostrar_tabla(df_turnos, width='stretch', hide_index=True)

                        if sim_seleccionada_global is None:
                            # Formato largo con una fila por (Tunel, Turno, Percentil) para poder
                            # graficar varios percentiles a la vez, uno por cada linea.
                            df_melt = pd.DataFrame(datos_turno_percentiles)
                            nombres_percentiles = " / ".join(percentiles_turno_sel)
                            titulo_grafico_turno = f'Avance Acumulado por Turno ({nombres_percentiles}) - {st.session_state.sistema_turnos}'
                            etiqueta_y_turno = 'Avance Acumulado (m)'

                            fig_turnos = px.line(
                                df_melt.sort_values(by='Turno_Num'),
                                x='Turno_Num', y='Avance Acumulado (m)',
                                color='Tunel',
                                line_dash='Percentil' if len(percentiles_turno_sel) > 1 else None,
                                title=titulo_grafico_turno,
                                labels={'Turno_Num': 'Número de Turno', 'Avance Acumulado (m)': etiqueta_y_turno},
                                markers=True,
                                hover_data={'Percentil': True}
                            )
                        else:
                            df_melt = df_turnos.melt(id_vars='Tunel', var_name='Turno', value_name='Avance Acumulado (m)')
                            df_melt['Turno_Num'] = df_melt['Turno'].apply(lambda x: int(x.split(' ')[0][1:].replace('T', '')))
                            # Se eliminan los turnos posteriores a la finalizacion del tunel para que
                            # la linea del grafico termine ahi, en vez de mostrar una asintota plana.
                            df_melt = df_melt.dropna(subset=['Avance Acumulado (m)'])

                            titulo_grafico_turno = f'Avance Acumulado por Turno - Simulación N° {sim_seleccionada_global} - {st.session_state.sistema_turnos}'
                            etiqueta_y_turno = f'Avance Acumulado Sim. {sim_seleccionada_global} (m)'

                            fig_turnos = px.line(
                                df_melt.sort_values(by='Turno_Num'),
                                x='Turno_Num', y='Avance Acumulado (m)',
                                color='Tunel',
                                title=titulo_grafico_turno,
                                labels={'Turno_Num': 'Número de Turno', 'Avance Acumulado (m)': etiqueta_y_turno},
                                markers=True
                            )
                        # El eje X parte en 0 (sin autorango, que dejaba turnos negativos) y termina en el último turno.
                        _max_turno_x = int(df_melt['Turno_Num'].max()) if not df_melt.empty else 1
                        fig_turnos.update_layout(xaxis=dict(
                            tickmode='linear', tick0=0, dtick=1, range=[0, _max_turno_x + 0.5]
                        ))
                        st.plotly_chart(fig_turnos, width='stretch')
            

            # ======================================================================
            # #####Menu 17b — Resultados: pestaña Recursos y Flota#####
            # ======================================================================
            with tab_recursos, seccion_menu('17b', 'Recursos y Flota'):

                if sim_seleccionada_global is None:
                    subtitulo_menu("Utilizacion de Recursos")
                else:
                    subtitulo_menu(f"Utilizacion de Recursos — Simulación N° {sim_seleccionada_global}")
                datos_recursos = []
                if st.session_state.estadisticas_recursos:
                    for recurso, stats_data in st.session_state.estadisticas_recursos.items():
                        if sim_seleccionada_global is None:
                            valor_utilizacion = np.mean(stats_data['utilizacion'])
                            valor_trabajo = np.mean(stats_data['tiempo_trabajando'])
                            valor_viaje = np.mean(stats_data['tiempo_viaje'])
                            col_uso, col_trabajo, col_viaje = (
                                'Uso Promedio (%)', 'Tiempo Productivo Promedio (h)', 'Tiempo de Viaje Promedio (h)'
                            )
                        else:
                            idx_sim = min(sim_seleccionada_global - 1, len(stats_data['utilizacion']) - 1)
                            valor_utilizacion = stats_data['utilizacion'][idx_sim]
                            valor_trabajo = stats_data['tiempo_trabajando'][idx_sim]
                            valor_viaje = stats_data['tiempo_viaje'][idx_sim]
                            col_uso, col_trabajo, col_viaje = (
                                f'Uso Sim. {sim_seleccionada_global} (%)',
                                f'Tiempo Productivo Sim. {sim_seleccionada_global} (h)',
                                f'Tiempo de Viaje Sim. {sim_seleccionada_global} (h)'
                            )

                        datos_recursos.append({
                            'Recurso': recurso,
                            'Cantidad': st.session_state.recursos_config[recurso]['cantidad'],
                            col_uso: round(valor_utilizacion, 2),
                            col_trabajo: round(valor_trabajo, 2),
                            col_viaje: round(valor_viaje, 2)
                        })
            
                df_recursos_res = pd.DataFrame(datos_recursos) if datos_recursos else None
                dashboard_tiempo_recursos(sim_seleccionada_global, df_recursos_res)

                # ----------------------------------------------------
                # Vista 3D del viaje de los recursos, por avance
                # ----------------------------------------------------
                menu_13b_viajes_3d(sim_seleccionada_global)


                # ----------------------------------------------------
                # Validacion del impacto de la Orquestacion de Flota por Turno
                # ----------------------------------------------------
                if st.session_state.get('registro_cambios_flota'):
                    with st.expander("Validar impacto de los cambios de flota por turno u hora", expanded=True):
                        st.markdown("#### Cambios de flota aplicados en esta simulación")
                        df_cambios_aplicados = pd.DataFrame(st.session_state.registro_cambios_flota).rename(columns={
                            'recurso': 'Recurso', 'turno': 'Turno', 'hora_inicio': 'Hora de Inicio (h)',
                            'definido_por': 'Definido por', 'tipo': 'Tipo de Cambio',
                            'cantidad_anterior': 'Cantidad Anterior', 'cantidad_nueva': 'Cantidad Nueva',
                            'demora_horas': 'Demora Aplicada (h)', 'tiempo_efectivo': 'Hora Efectiva del Cambio'
                        })
                        mostrar_tabla(df_cambios_aplicados, width='stretch', hide_index=True)
                        ayuda_inline("Registro informativo de los cambios de flota aplicados en la simulación ejecutada.", "Sobre esta tabla")




            # ----------------------------------------------------
            # Carta Gantt y Tiempos Detallados
            # ----------------------------------------------------

            # ======================================================================
            # #####Menu 18 — Carta Gantt y Tiempos Detallados#####
            # ======================================================================
            with tab_gantt, seccion_menu('18', 'Carta Gantt y Tiempos Detallados'):

                # Cambios de flota por turno: son globales al recurso (no a un tunel especifico),
                # por lo que se muestran aparte, con su demora de activacion/desactivacion aplicada.
                df_fleet_changes = st.session_state.df_traza_eventos_cache
                if not df_fleet_changes.empty and 'type' in df_fleet_changes.columns:
                    df_fleet_changes = df_fleet_changes[df_fleet_changes['type'] == 'fleet_change']
                    if not df_fleet_changes.empty:
                        with st.expander("Línea de tiempo de cambios de flota (con demora aplicada)"):
                            df_fc_sim1 = df_fleet_changes[df_fleet_changes['Simulation_ID'] == 1].sort_values('Start')
                            cols_fc = ['Recurso', 'Actividad', 'Turno_Plan'] + [c for c in ('Hora_Plan', 'Definido_Por') if c in df_fc_sim1.columns] + ['Start', 'Demora_Aplicada_h']
                            df_fc_display = df_fc_sim1[cols_fc].rename(columns={
                                'Actividad': 'Cambio', 'Turno_Plan': 'Turno Configurado',
                                'Hora_Plan': 'Hora de Inicio (h)', 'Definido_Por': 'Definido por',
                                'Start': 'Hora Efectiva (h)', 'Demora_Aplicada_h': 'Demora Aplicada (h)'
                            })
                            mostrar_tabla(df_fc_display, width='stretch', hide_index=True)
            
                df_traza_completa = st.session_state.df_traza_eventos_cache
            
                if df_traza_completa.empty:
                    st.warning("No hay datos de túneles en la traza.")
                    tunel_seleccionado = None
                else:
                    frentes_con_traza = sorted(df_traza_completa['Resource_Origin_Tunnel'].unique())
                    tunel_seleccionado = st.selectbox("Seleccione Túnel para el Detalle", frentes_con_traza)

                if tunel_seleccionado:
                    df_tunel = df_traza_completa[df_traza_completa['Resource_Origin_Tunnel'] == tunel_seleccionado].copy()
                    n_sims_real = df_tunel['Simulation_ID'].nunique() if not df_tunel.empty else 1
                    # Las mantenciones programadas de los equipos que uso este tunel se tratan como una
                    # actividad mas (Resumen de Tiempos, Carta Gantt, distribucion Monte Carlo).
                    df_tunel = incorporar_mantenciones_como_actividad(df_tunel, df_traza_completa, tunel_seleccionado)
                    # Para la opcion 'Incluir viajes' de la Carta Gantt (viajes cuyo destino es este tunel).
                    df_tunel_gantt = anexar_viajes_destino(df_tunel, df_traza_completa, tunel_seleccionado)

                    mostrar_restricciones_del_tunel(tunel_seleccionado)

                    tiempo_limite_ui = float(st.session_state.get(
                        'tiempo_limite_simulado',
                        st.session_state.get('numero_turnos_simulado', 62) * st.session_state.get('duracion_turno_simulada', 8.0)
                    ))
                    duracion_turno_ui = float(st.session_state.get('duracion_turno_simulada', 8.0))
                    n_sims_ui = int(st.session_state.get('n_simulaciones_ejecutadas', n_sims_real))

                    # ---- Selector: resumen del total del tunel o de un avance puntual ----
                    # Aplica a las dos tablas de abajo (tiempos y viajes). Siempre usa todas las
                    # simulaciones (o la simulacion elegida arriba, si hay una).
                    run_id_res = st.session_state.run_id
                    ciclos_res, sims_por_avance_res, avances_prom_res = info_avances_tunel(
                        df_tunel, run_id_res, tunel_seleccionado, tiempo_limite_ui
                    )
                    avance_res_sel = st.selectbox(
                        "Ver resumen de",
                        [0] + ciclos_res,
                        format_func=lambda a: formatear_opcion_avance(a, sims_por_avance_res, avances_prom_res),
                        key=f"avance_resumen_sel_{tunel_seleccionado}",
                        help="'Total' suma todos los avances que logró el túnel. 'Avance k' muestra solo lo ocurrido "
                             "en ese avance (promedio de las simulaciones que lo alcanzaron)."
                    )
                    avance_res = None if avance_res_sel == 0 else int(avance_res_sel)
                    sufijo_avance_res = '' if avance_res is None else f" — Avance {avance_res}"

                    col_gantt, col_viaje = st.columns(2)
                
                    with col_gantt:
                        if sim_seleccionada_global is None:
                            st.markdown(f"#### Resumen de Tiempos por Actividad/Demora (Horas){sufijo_avance_res}")
                        else:
                            st.markdown(f"#### Resumen de Tiempos por Actividad/Demora (Horas) — Sim. {sim_seleccionada_global}{sufijo_avance_res}")
                        df_ciclo_events = df_tunel[df_tunel['type'].isin(['activity', 'delay', 'delay_fh_oc', 'delay_res', 'delay_bloqueo_total', 'delay_equipment_failure', 'delay_fleet_change', 'delay_maintenance', 'delay_shift_change', 'blocked_no_resource'] + TIPOS_MANTENCION)].copy()
                    
                        if not df_ciclo_events.empty:
                            # Una fila por PASO (Secuencia + Actividad), en orden de secuencia: asi las
                            # actividades repetidas (ej. FR en las secuencias 7, 10 y 15) ya no se fusionan.
                            tiempos_resumen = calcular_resumen_tiempos_pasos(
                                df_tunel, st.session_state.run_id, tunel_seleccionado, sim_seleccionada_global,
                                ciclo=avance_res
                            )
                            if tiempos_resumen.empty:
                                if avance_res is None:
                                    st.info(f"La simulación N° {sim_seleccionada_global} no registró eventos de este tipo en este túnel.")
                                elif sim_seleccionada_global is None:
                                    st.info(f"No hay eventos registrados en el avance {avance_res} de este túnel.")
                                else:
                                    st.info(f"La simulación N° {sim_seleccionada_global} no llegó al avance {avance_res} de este túnel.")
                            else:
                                col_tiempo_tabla = (
                                    'Tiempo Total Promedio (h)' if sim_seleccionada_global is None
                                    else f'Tiempo Total Sim. {sim_seleccionada_global} (h)'
                                )
                                if avance_res is not None:
                                    col_tiempo_tabla = col_tiempo_tabla.replace('Total', f'Avance {avance_res}')
                                tiempos_resumen = tiempos_resumen.rename(columns={'Duration': col_tiempo_tabla})
                                tiempos_resumen[col_tiempo_tabla] = r2(tiempos_resumen[col_tiempo_tabla])
                                mostrar_tabla(tiempos_resumen, width='stretch', hide_index=True)
                                ayuda_inline(
                                    ("Tiempo de cada paso sumando todos los avances del túnel. "
                                     if avance_res is None else
                                     f"Tiempo de cada paso dentro del avance {avance_res} (promedio de las simulaciones que lo alcanzaron). ") +
                                    "Si una secuencia usa 2 recursos en paralelo, cada recurso aparece en su propia fila con su propio tiempo.",
                                    "Cómo leer la tabla")
                        else:
                            st.info("No hay eventos operacionales o demoras para este túnel.")

                    with col_viaje:
                        if sim_seleccionada_global is None:
                            st.markdown(f"#### Resumen de Viajes (Recursos){sufijo_avance_res}")
                        else:
                            st.markdown(f"#### Resumen de Viajes (Recursos) — Sim. {sim_seleccionada_global}{sufijo_avance_res}")
                        if avance_res is None:
                            df_viajes = df_tunel[df_tunel['type'] == 'travel'].copy()
                            viajes_agregados = (
                                calcular_viajes_agregados(
                                    df_tunel, st.session_state.run_id, tunel_seleccionado, sim_seleccionada_global
                                ) if not df_viajes.empty else None
                            )
                        else:
                            viajes_agregados = calcular_viajes_por_avance(
                                df_traza_completa, st.session_state.run_id, tunel_seleccionado, avance_res
                            )
                            if viajes_agregados.empty:
                                viajes_agregados = None

                        if viajes_agregados is not None:

                            if sim_seleccionada_global is None:
                                viajes_summary = viajes_agregados.drop(columns='Simulation_ID').groupby('Recurso_Tipo').mean(numeric_only=True).reset_index()
                                viajes_summary.rename(columns={
                                    'Recurso_Tipo': 'Recurso',
                                    'Distancia_Total': 'Distancia Total Promedio (m/sim)',
                                    'Tiempo_Total': 'Tiempo Total Promedio (h/sim)',
                                    'Num_Viajes': 'Número de Viajes Promedio (viajes/sim)'
                                }, inplace=True)
                            else:
                                viajes_summary = viajes_agregados[
                                    viajes_agregados['Simulation_ID'] == sim_seleccionada_global
                                ].drop(columns='Simulation_ID').reset_index(drop=True)
                                viajes_summary.rename(columns={
                                    'Recurso_Tipo': 'Recurso',
                                    'Distancia_Total': f'Distancia Total Sim. {sim_seleccionada_global} (m)',
                                    'Tiempo_Total': f'Tiempo Total Sim. {sim_seleccionada_global} (h)',
                                    'Num_Viajes': f'Número de Viajes Sim. {sim_seleccionada_global}'
                                }, inplace=True)

                            if viajes_summary.empty:
                                if avance_res is None:
                                    st.info(f"La simulación N° {sim_seleccionada_global} no registró viajes de recursos en este túnel.")
                                else:
                                    st.info(f"La simulación N° {sim_seleccionada_global} no registró viajes hacia este túnel en el avance {avance_res}.")
                            else:
                                if avance_res is not None:
                                    viajes_summary.columns = [c.replace('Total', f'Avance {avance_res}') for c in viajes_summary.columns]
                                cols_num_viajes = viajes_summary.select_dtypes(include=[np.number]).columns.tolist()
                                viajes_summary[cols_num_viajes] = viajes_summary[cols_num_viajes].apply(r2)
                                mostrar_tabla(
                                    viajes_summary, 
                                    width='stretch', 
                                    hide_index=True
                                )
                                if avance_res is not None:
                                    ayuda_inline(f"Viajes de los recursos para llegar a este túnel durante su avance {avance_res}. El total del túnel (sin elegir avance) cuenta los viajes según el túnel de origen del recurso, por lo que no es la suma de los avances.", "Sobre estos viajes")
                        else:
                            if avance_res is None:
                                st.info("No se registraron viajes de recursos para este túnel.")
                            else:
                                st.info(f"No se registraron viajes hacia este túnel en el avance {avance_res}.")

                    st.markdown("---")
                
                    st.markdown("#### Carta Gantt de Actividades")

                    VISTA_TODOS = "Todos los avances del túnel (tiempo por actividad)"
                    VISTA_CICLO = "Por avance — una simulación (Gantt real, con horas reales)"
                    VISTA_PROB = "Por avance — probabilística (todas las simulaciones)"
                    # Los dos mundos quedan separados por el "Modo de visualización de resultados":
                    #  - Promedio (sim_seleccionada_global is None): vistas probabilísticas (todas las simulaciones).
                    #  - Simulación específica: vistas de ESA simulación (usa la simulación elegida arriba).
                    if sim_seleccionada_global is None:
                        opciones_vista_gantt = [VISTA_TODOS, VISTA_PROB]
                        ayuda_vista_gantt = (
                            "Modo Todas las Simulaciones:\n"
                            "• Todos los avances: cada actividad (en orden de secuencia) con su tiempo sumando todos "
                            "los avances que alcanzó el túnel, con percentiles entre simulaciones.\n"
                            "• Por avance (probabilística): eliges el avance y ves la duración de cada actividad "
                            "con percentiles considerando todas las simulaciones.\n"
                            "Para ver una simulación puntual, cambia el Modo de visualización a 'Simulación específica'."
                        )
                        key_vista_gantt = "vista_gantt_radio_promedio"
                    else:
                        opciones_vista_gantt = [VISTA_TODOS, VISTA_CICLO]
                        ayuda_vista_gantt = (
                            f"Modo Simulación específica (N° {sim_seleccionada_global}):\n"
                            "• Todos los avances: tiempo exacto de cada actividad sumando los avances de esa simulación.\n"
                            "• Por avance (una simulación): eliges el avance (1, 2, 3... incluido el avance parcial en que "
                            "quedó el túnel al acabarse el tiempo) y ves cuándo partió y terminó cada actividad.\n"
                            "Para ver percentiles entre simulaciones, cambia el Modo de visualización a 'Todas las Simulaciones'."
                        )
                        key_vista_gantt = "vista_gantt_radio_simulacion"
                    vista_gantt = st.radio(
                        "Vista de la Carta Gantt",
                        opciones_vista_gantt,
                        key=key_vista_gantt,
                        help=ayuda_vista_gantt
                    )

                    df_ciclo_events = df_tunel[df_tunel['type'].isin(TIPOS_EVENTO_CICLO_GANTT)].copy()
                    run_id_ui = st.session_state.run_id

                    if df_ciclo_events.empty:
                        if sim_seleccionada_global is not None:
                            st.info(f"La simulación N° {sim_seleccionada_global} no registró actividades/demoras para este túnel.")

                    # ------------------ Vista 1: todos los avances ------------------
                    elif vista_gantt == VISTA_TODOS:
                        if sim_seleccionada_global is not None:
                            ayuda_inline("Al ver una simulación específica, la carta deja de ser probabilística: cada barra muestra la duración exacta ocurrida, sin rango P10-P90.", "Modo simulación específica")
                        modo_gantt = 'acumulado'  # único modo: tiempo acumulado de todos los avances del túnel
                        col_gantt_pct = st.container()
                        with col_gantt_pct:
                            if sim_seleccionada_global is None:
                                percentil_gantt_label = st.selectbox(
                                    "Percentil a usar como largo de la barra / eje acumulado",
                                    list(PERCENTILES_DISPONIBLES_GANTT.keys()),
                                    index=list(PERCENTILES_DISPONIBLES_GANTT.keys()).index('P50 (Mediana)'),
                                    key="percentil_gantt_select",
                                    help="Percentil entre simulaciones que define el largo de cada barra. "
                                         "El rango gris (P10–P90) se muestra siempre como referencia."
                                )
                                percentil_gantt_col = PERCENTILES_DISPONIBLES_GANTT[percentil_gantt_label]
                            else:
                                percentil_gantt_col = 'P50'
                                ayuda_inline("En modo simulación específica se muestra el valor exacto ocurrido.", "Valor mostrado")

                        incluir_viajes_todos = st.checkbox(
                            "Incluir viajes de recursos", value=False, key="gantt_todos_incluir_viajes",
                            help="Agrega, antes de cada actividad, el viaje del recurso hacia este túnel "
                                 "(el tiempo de la actividad de ese recurso parte cuando termina su viaje)."
                        )
                        gantt_data = calcular_pasos_gantt(
                            df_tunel_gantt, run_id_ui, tunel_seleccionado, sim_seleccionada_global,
                            modo=modo_gantt, percentil_barra=percentil_gantt_col,
                            incluir_viajes=incluir_viajes_todos
                        )
                        if gantt_data:
                            etiqueta_modo = "acumulado"
                            etiqueta_titulo_modo = "Tiempo por Actividad (todos los avances)"
                            if sim_seleccionada_global is None:
                                titulo_gantt = f'Carta Gantt (Túnel: {tunel_seleccionado}) - {etiqueta_titulo_modo} [{percentil_gantt_col}, con Rango P10-P90]'
                            else:
                                titulo_gantt = f'Carta Gantt (Túnel: {tunel_seleccionado}) - Simulación N° {sim_seleccionada_global} ({etiqueta_titulo_modo})'
                            eje_x_titulo = 'Horas acumuladas'
                            st.plotly_chart(
                                figura_gantt_cascada(gantt_data, titulo_gantt, eje_x_titulo,
                                                     sim_seleccionada_global is not None, percentil_gantt_col, etiqueta_modo),
                                width='stretch'
                            )
                            st.caption(f"{len(gantt_data)} barras, en orden de secuencia", help="Las actividades repetidas (ej. FR) aparecen una vez por cada secuencia en que están. Si una secuencia usa recursos en paralelo, cada recurso tiene su propia barra; parten juntas y la siguiente secuencia arranca cuando termina la más larga. Las demoras van primero. El largo de cada barra es el percentil elegido y el bigote gris marca el rango P10–P90 (solo al ver todas las simulaciones).")
                        elif sim_seleccionada_global is not None:
                            st.info(f"La simulación N° {sim_seleccionada_global} no registró actividades/demoras para este túnel.")

                    # ------------------ Vista 2: un avance, una simulación ------------------
                    elif vista_gantt == VISTA_CICLO:
                        estado_ciclos = calcular_estado_ciclos_tunel(df_tunel, run_id_ui, tunel_seleccionado, tiempo_limite_ui)
                        if estado_ciclos.empty:
                            st.info("Presione **SIMULAR** de nuevo para generar los datos por avance (o este túnel no ejecutó ningún avance).")
                        else:
                            # Esta vista solo existe en modo "Simulación específica": usa la simulación elegida arriba.
                            sim_gantt = sim_seleccionada_global
                            st.caption(f"Usando la simulación N° {sim_gantt} elegida arriba (Modo de visualización).")
                            est_sim = estado_ciclos[estado_ciclos['Simulation_ID'] == sim_gantt].sort_values('Ciclo_Frente')
                            if est_sim.empty:
                                st.info(f"En la simulación N° {sim_gantt} el túnel {tunel_seleccionado} no alcanzó a ejecutar ningún avance.")
                            else:
                                n_completos = int(est_sim['Completo'].sum())
                                avances_equiv = float(est_sim['Fraccion'].sum())
                                parcial = est_sim[~est_sim['Completo']]
                                resumen_av = f"{n_completos} avance(s) completo(s)"
                                if not parcial.empty:
                                    resumen_av += f" + {parcial.iloc[-1]['Fraccion'] * 100:.0f}% del avance {int(parcial.iloc[-1]['Ciclo_Frente'])}"
                                st.markdown(
                                    f"En la simulación **N° {sim_gantt}**, el túnel **{tunel_seleccionado}** alcanzó "
                                    f"**{avances_equiv:.1f} avances** ({resumen_av})."
                                )
                                ayuda_inline("El % del avance parcial es el porcentaje de las secuencias de ese avance que alcanzaron a terminar antes de acabarse el tiempo simulado.", "Avance parcial")

                                def _fmt_ciclo(k, est_sim=est_sim):
                                    fila = est_sim[est_sim['Ciclo_Frente'] == k].iloc[0]
                                    dur = fila['Fin'] - fila['Inicio']
                                    if fila['Completo']:
                                        return f"Avance {k} — completo ({fmt2(dur)} h)"
                                    return f"Avance {k} — parcial (~{fila['Fraccion'] * 100:.0f}% hecho)"

                                col_av, col_pct_ref, col_viajes = st.columns([2, 1.3, 1])
                                with col_av:
                                    ciclo_sel = st.selectbox(
                                        "Avance a visualizar", list(est_sim['Ciclo_Frente']),
                                        format_func=_fmt_ciclo, key="ciclo_gantt_sel"
                                    )
                                with col_pct_ref:
                                    pct_ref_label = st.selectbox(
                                        "Percentil de referencia (al pasar el mouse)",
                                        ["Sin referencia"] + list(PERCENTILES_DISPONIBLES_GANTT.keys()),
                                        index=1 + list(PERCENTILES_DISPONIBLES_GANTT.keys()).index('P50 (Mediana)'),
                                        key="pct_ref_gantt_ciclo",
                                        help="Agrega al tooltip de cada actividad el percentil elegido de la duración de esa misma "
                                             "actividad en este avance (calculado con todas las simulaciones que lo completaron) y en "
                                             "qué percentil quedó esta simulación. No dibuja nada sobre la carta."
                                    )
                                with col_viajes:
                                    st.write("")
                                    incluir_viajes = st.checkbox("Incluir viajes de recursos", value=False, key="gantt_incluir_viajes")

                                completos_av = estado_ciclos[(estado_ciclos['Ciclo_Frente'] == ciclo_sel) & estado_ciclos['Completo']]
                                ref_pasos, col_ref = None, None
                                n_sims_ref = int(completos_av['Simulation_ID'].nunique())
                                if pct_ref_label != "Sin referencia" and n_sims_ref > 0:
                                    col_ref = PERCENTILES_DISPONIBLES_GANTT[pct_ref_label]
                                    sims_av = list(completos_av['Simulation_ID'])
                                    df_ev_av = preparar_eventos_ciclo_tunel(df_tunel, run_id_ui, tunel_seleccionado)
                                    df_ev_av = df_ev_av[df_ev_av['Simulation_ID'].isin(sims_av)
                                                        & (df_ev_av['Ciclo_Frente'] == ciclo_sel)]
                                    if not df_ev_av.empty:
                                        spans_av = _spans_por_paso_ciclo(df_ev_av)
                                        ref_pasos = {
                                            (int(k[0]), str(k[1]), str(k[2])): g['Duration'].values.astype(float)
                                            for k, g in spans_av.groupby(['Secuencia', 'Actividad', 'Recurso_Tipo'])
                                        }

                                fila_ciclo = est_sim[est_sim['Ciclo_Frente'] == ciclo_sel].iloc[0]
                                df_ev_ciclo = obtener_eventos_ciclo_real(
                                    df_traza_completa, run_id_ui, tunel_seleccionado, sim_gantt, ciclo_sel, incluir_viajes
                                )
                                if df_ev_ciclo.empty:
                                    st.info("No hay eventos registrados para ese avance.")
                                else:
                                    inicio_ciclo = float(df_ev_ciclo['Start'].min())
                                    limite_rel = None if fila_ciclo['Completo'] else (tiempo_limite_ui - inicio_ciclo)
                                    titulo_ciclo = (
                                        f"Carta Gantt (Túnel: {tunel_seleccionado}) — Sim. {sim_gantt}, Avance {ciclo_sel} "
                                        f"({'completo' if fila_ciclo['Completo'] else 'parcial'})"
                                    )
                                    st.plotly_chart(
                                        figura_gantt_ciclo_real(df_ev_ciclo, inicio_ciclo, titulo_ciclo, duracion_turno_ui,
                                                                limite_rel, ref_pasos, col_ref),
                                        width='stretch'
                                    )
                                    st.caption(f"Avance iniciado en la hora {fmt2(inicio_ciclo)}", help="Una fila por actividad y recurso, con sus horas reales de inicio y fin.")
                                    if limite_rel is not None and limite_rel > 0:
                                        st.caption(
                                            f"Avance incompleto: el horizonte simulado terminó a las {fmt2(limite_rel)} h "
                                            "desde el inicio de este avance."
                                        )
                                    if ref_pasos:
                                        st.caption(f"Referencia: {col_ref} ({n_sims_ref} simulaciones)", help=f"Pase el mouse sobre una actividad para ver el {col_ref} de su duración en este avance y en qué percentil quedó esta simulación. Las demoras y los viajes no llevan referencia.")
                                    elif pct_ref_label != "Sin referencia":
                                        st.caption("Ninguna simulación completó este avance, así que no hay percentil de referencia.")

                                with st.expander("Ver todos los avances de esta simulación"):
                                    df_av_tabla = est_sim.copy()
                                    df_av_tabla['Estado'] = df_av_tabla['Completo'].map({True: 'Completo', False: 'Parcial'})
                                    df_av_tabla['Duración (h)'] = r2(df_av_tabla['Fin'] - df_av_tabla['Inicio'])
                                    df_av_tabla['Inicio (h)'] = r2(df_av_tabla['Inicio'])
                                    df_av_tabla['Fin (h)'] = r2(df_av_tabla['Fin'])
                                    df_av_tabla['% hecho'] = r2(df_av_tabla['Fraccion'] * 100)
                                    mostrar_tabla(
                                        df_av_tabla.rename(columns={'Ciclo_Frente': 'Avance'})[
                                            ['Avance', 'Estado', '% hecho', 'Inicio (h)', 'Fin (h)', 'Duración (h)']
                                        ], width='stretch', hide_index=True
                                    )

                    # ------------------ Vista 3: un avance, todas las simulaciones ------------------
                    else:
                        estado_ciclos = calcular_estado_ciclos_tunel(df_tunel, run_id_ui, tunel_seleccionado, tiempo_limite_ui)
                        completos = estado_ciclos[estado_ciclos['Completo']] if not estado_ciclos.empty else estado_ciclos
                        if completos.empty:
                            st.info("Presione **SIMULAR** de nuevo para generar los datos por avance (o ninguna simulación completó un avance en este túnel).")
                        else:
                            conteo_ciclos = completos.groupby('Ciclo_Frente')['Simulation_ID'].nunique()
                            n_sims_con_avances = int(estado_ciclos['Simulation_ID'].nunique())

                            col_av_p, col_pct_p = st.columns(2)
                            with col_av_p:
                                ciclo_prob = st.selectbox(
                                    "Avance a visualizar", list(conteo_ciclos.index),
                                    format_func=lambda k: f"Avance {k} ({int(conteo_ciclos[k])} de {n_sims_ui} simulaciones lo completaron)",
                                    key="ciclo_gantt_prob_sel"
                                )
                            with col_pct_p:
                                pct_label_prob = st.selectbox(
                                    "Percentil a usar como largo de la barra",
                                    list(PERCENTILES_DISPONIBLES_GANTT.keys()),
                                    index=list(PERCENTILES_DISPONIBLES_GANTT.keys()).index('P50 (Mediana)'),
                                    key="percentil_gantt_prob_select"
                                )
                                pct_col_prob = PERCENTILES_DISPONIBLES_GANTT[pct_label_prob]

                            sims_ciclo = tuple(sorted(completos[completos['Ciclo_Frente'] == ciclo_prob]['Simulation_ID'].tolist()))
                            incluir_viajes_prob = st.checkbox(
                                "Incluir viajes de recursos", value=False, key="gantt_prob_incluir_viajes",
                                help="Agrega, antes de cada actividad, el viaje del recurso hacia este túnel "
                                     "(el tiempo de la actividad de ese recurso parte cuando termina su viaje)."
                            )
                            gantt_prob = calcular_pasos_gantt(
                                df_tunel_gantt, run_id_ui, tunel_seleccionado, None, percentil_barra=pct_col_prob,
                                ciclo=ciclo_prob, sims_incluidas=sims_ciclo, incluir_viajes=incluir_viajes_prob
                            )
                            if gantt_prob:
                                titulo_prob = (
                                    f'Carta Gantt Probabilística (Túnel: {tunel_seleccionado}) — Avance {ciclo_prob} '
                                    f'[{pct_col_prob}, rango P10-P90, {len(sims_ciclo)} simulaciones]'
                                )
                                st.plotly_chart(
                                    figura_gantt_cascada(gantt_prob, titulo_prob, 'Horas acumuladas dentro del avance',
                                                         False, pct_col_prob, f"Avance {ciclo_prob}"),
                                    width='stretch'
                                )
                                sub_ciclo = completos[completos['Ciclo_Frente'] == ciclo_prob]
                                dur_real = (sub_ciclo['Fin'] - sub_ciclo['Inicio']).values.astype(float)
                                avances_por_sim = estado_ciclos.groupby('Simulation_ID')['Fraccion'].sum().values.astype(float)
                                st.caption(f"Avance {ciclo_prob}: P10 {fmt2(np.percentile(dur_real, 10))} h | P50 {fmt2(np.percentile(dur_real, 50))} h | P90 {fmt2(np.percentile(dur_real, 90))} h. Avances por simulación: P10 {fmt2(np.percentile(avances_por_sim, 10))} | P50 {fmt2(np.percentile(avances_por_sim, 50))} | P90 {fmt2(np.percentile(avances_por_sim, 90))}", help="Duración real del avance (incluye esperas por recursos). Las barras muestran el tiempo de cada actividad; solo se consideran las simulaciones que completaron ese avance.")
                            else:
                                st.info("No hay datos para ese avance.")

                    st.markdown("---")

                    # ----------------------------------------------------
                    # 14.1 Distribucion de Resultados Monte Carlo por Actividad/Demora
                    # ----------------------------------------------------
                    st.markdown("#### Distribución de Resultados Monte Carlo por Actividad/Demora")
                    with st.expander("Cómo leer esta sección", expanded=False):
                        ayuda_inline("Cada paso (secuencia + actividad) aparece por separado; si usa recursos en paralelo, cada recurso tiene su propia distribución. Puede ver el total de avances del túnel o un avance puntual, con todas las simulaciones. Se usan todas las ocurrencias, no solo el promedio, y se superpone la distribución de entrada si existe.", "Cómo leer esta vista")

                    df_resultados_mc = df_tunel[df_tunel['type'].isin(
                        ['activity', 'delay', 'delay_fh_oc', 'delay_res', 'delay_bloqueo_total', 'delay_equipment_failure', 'delay_fleet_change', 'delay_maintenance', 'delay_shift_change'] + TIPOS_MANTENCION
                    )].copy()

                    if df_resultados_mc.empty or 'Actividad' not in df_resultados_mc.columns:
                        st.info("No hay resultados de actividades o demoras para graficar en este tunel.")
                    else:
                        df_mc, claves_mc, etiquetas_mc = preparar_muestras_mc(df_tunel)
                        valores_mc = np.array([], dtype=float)
                        actividad_mc_sel, titulo_mc, clave_mc_sel, texto_avance_mc = '', '', None, ''

                        if claves_mc:
                            col_paso_mc, col_avance_mc = st.columns([3, 2])
                            with col_paso_mc:
                                clave_mc_sel = st.selectbox(
                                    "Seleccione Actividad o Demora (cada recurso en paralelo por separado)",
                                    claves_mc,
                                    format_func=lambda k: etiquetas_mc[k],
                                    key=f"paso_mc_sel_{tunel_seleccionado}"
                                )
                            actividad_mc_sel = etiquetas_mc[clave_mc_sel]

                            # Avances del tunel (Ciclo_Frente >= 1) y cuantas simulaciones llegaron a cada uno.
                            ciclos_mc = sorted(int(c) for c in df_mc['Ciclo_Frente'].unique() if c >= 1)
                            sims_por_avance_mc = df_mc[df_mc['Ciclo_Frente'] >= 1].groupby('Ciclo_Frente')['Simulation_ID'].nunique()
                            # Avances totales logrados por simulacion (ej. 4.6 = 4 completos + 60% del quinto).
                            estado_mc = calcular_estado_ciclos_tunel(df_tunel, run_id_ui, tunel_seleccionado, tiempo_limite_ui)
                            avances_prom_mc = (
                                float(estado_mc.groupby('Simulation_ID')['Fraccion'].sum().mean())
                                if not estado_mc.empty else None
                            )

                            def _fmt_avance_mc(a):
                                if a == 0:
                                    extra = f" (~{fmt2(avances_prom_mc)} avances por simulación)" if avances_prom_mc is not None else ""
                                    return "Total — todos los avances del túnel" + extra
                                return f"Avance {a} ({int(sims_por_avance_mc.get(a, 0))} simulaciones)"

                            with col_avance_mc:
                                avance_mc_sel = st.selectbox(
                                    "Avance a analizar", [0] + ciclos_mc,
                                    format_func=_fmt_avance_mc,
                                    key=f"avance_mc_sel_{tunel_seleccionado}"
                                )
                            avance_mc = None if avance_mc_sel == 0 else int(avance_mc_sel)
                            texto_avance_mc = "Total de avances" if avance_mc is None else f"Avance {avance_mc}"
                            titulo_mc = f"{actividad_mc_sel} — {texto_avance_mc}"
                            valores_mc = muestras_paso_mc(df_mc, clave_mc_sel, avance_mc)

                        if len(valores_mc) == 0:
                            st.info("No hay valores de duracion para este paso/recurso en el avance seleccionado de este tunel.")
                        else:
                            pct_mc = percentiles_reporte(valores_mc)
                            n_trunc_mc = contar_truncados_horizonte(df_mc, clave_mc_sel, avance_mc)
                            if n_trunc_mc > 0:
                                st.caption(f"{n_trunc_mc} de {len(valores_mc)} muestras interrumpidas por el límite de tiempo", help="En la traza/Gantt se ven más cortas. Aquí se muestra su duración muestreada completa, que sí respeta el rango definido.")

                            # Una sola cuadrícula: Muestras y Esperanza primero, luego los percentiles de menor a mayor.
                            # Se destacan la Esperanza y la Mediana.
                            mostrar_tarjetas_metricas([
                                ("Muestras (N)", f"{len(valores_mc)}", "", False),
                                ("Esperanza", fmt2(pct_mc['esperanza']), "h", True),
                                ("P0 · Mín.", fmt2(pct_mc['p0']), "h", False),
                                ("P10", fmt2(pct_mc['p10']), "h", False),
                                ("P30", fmt2(pct_mc['p30']), "h", False),
                                ("P50 · Mediana", fmt2(pct_mc['p50']), "h", True),
                                ("P70", fmt2(pct_mc['p70']), "h", False),
                                ("P90", fmt2(pct_mc['p90']), "h", False),
                                ("P100 · Máx.", fmt2(pct_mc['p100']), "h", False),
                            ])

                            # Busca si la actividad tiene una distribucion de entrada definida
                            # (original o modificada) para superponerla con el resultado simulado
                            # Distribucion de entrada de ESE paso y ESE recurso (no la primera que
                            # coincida por nombre).
                            info_dist_teorica = buscar_distribucion_entrada(
                                st.session_state.get('df_actividades_simuladas', pd.DataFrame()), clave_mc_sel
                            )
                            if info_dist_teorica is None:
                                # Demora por Falla de Equipo: su distribucion de entrada vive en el
                                # escenario de falla (Menu 11), no en la hoja de Actividades.
                                info_dist_teorica = buscar_distribucion_falla(
                                    clave_mc_sel[1], tunel_seleccionado,
                                    st.session_state.get('fallas_equipos', [])
                                )
                            if info_dist_teorica is None:
                                # Mantencion programada: su distribucion de entrada (y rango truncado)
                                # vive en la configuracion del Menu 12.
                                info_dist_teorica = buscar_distribucion_mantencion(
                                    clave_mc_sel[1], st.session_state.get('mantenciones_programadas', [])
                                )

                            if info_dist_teorica is None:
                                info_dist_teorica = buscar_distribucion_cambio_turno(
                                    clave_mc_sel[1], st.session_state.get('demoras_cambio_turno')
                                )

                            # Rango de X común a los dos gráficos: desde 0 hasta un poco más del valor máximo observado.
                            x_max_rango = max(float(np.max(valores_mc)) * 1.15, 1e-6)
                            sin_dist_entrada = False

                            col_hist1, col_hist2 = st.columns(2)

                            # --- Gráfico 1: Función de Densidad (curva teórica continua + resultados
                            # de Monte Carlo como líneas verticales individuales, sin histograma) ---
                            with col_hist1:
                                x_vals = np.linspace(0, x_max_rango, 400)

                                fig_dens_h = go.Figure()

                                hay_dist_teorica = (
                                    info_dist_teorica is not None
                                    and str(info_dist_teorica.get('Distribucion', 'Cte')) != 'Cte'
                                    and isinstance(info_dist_teorica.get('Tiempo'), dict)
                                )

                                y_pdf_teorica = None
                                if hay_dist_teorica:
                                    y_pdf_teorica = calcular_pdf_teorica(
                                        info_dist_teorica['Tiempo'], info_dist_teorica['Distribucion'], x_vals
                                    )

                                # Altura de referencia para las lineas verticales (rug plot).
                                # Se ancla a la altura de la curva teorica si existe; si no, se usa
                                # una referencia generica basada en un KDE auxiliar (solo para escala,
                                # no se dibuja como curva).
                                if y_pdf_teorica is not None and np.max(y_pdf_teorica) > 0:
                                    altura_ref = float(np.max(y_pdf_teorica))
                                elif len(valores_mc) >= 2 and np.std(valores_mc) > 1e-9:
                                    altura_ref = float(np.max(stats.gaussian_kde(valores_mc)(x_vals)))
                                else:
                                    altura_ref = 1.0
                                altura_lineas = max(altura_ref * 0.12, 1e-6)

                                # Lineas verticales: una por cada resultado individual de la simulacion
                                rug_x, rug_y = [], []
                                for v in valores_mc:
                                    rug_x.extend([v, v, None])
                                    rug_y.extend([0, altura_lineas, None])

                                fig_dens_h.add_trace(go.Scatter(
                                    x=rug_x, y=rug_y,
                                    mode='lines',
                                    name='Resultado simulado (Monte Carlo)',
                                    line=dict(color='rgba(94, 122, 148, 0.55)', width=1),
                                    hoverinfo='skip'
                                ))

                                if hay_dist_teorica:
                                    fig_dens_h.add_trace(go.Scatter(
                                        x=x_vals, y=y_pdf_teorica,
                                        mode='lines',
                                        name=f"Distribución de entrada ({info_dist_teorica['Distribucion']})",
                                        line=dict(color='#F2A33A', width=2),
                                        fill='tozeroy',
                                        fillcolor='rgba(242, 163, 58, 0.30)'
                                    ))
                                else:
                                    sin_dist_entrada = True

                                estilo_figura_mc(
                                    fig_dens_h, f'Función de Densidad — {titulo_mc}', 'Duración (horas)',
                                    'Densidad de probabilidad', con_leyenda=True, rango_x=[0, x_max_rango]
                                )
                                st.plotly_chart(fig_dens_h, width='stretch')

                            # --- Gráfico 2: Histograma de CONTEO por intervalo de tiempo ---
                            with col_hist2:
                                fig_conteo = go.Figure()
                                fig_conteo.add_trace(go.Histogram(
                                    x=valores_mc,
                                    name='N° de Ocurrencias (Monte Carlo)',
                                    marker_color='#5E7A94',
                                    opacity=0.9
                                ))

                                estilo_figura_mc(
                                    fig_conteo, f'N° de Veces por Intervalo de Tiempo — {titulo_mc}', 'Duración (horas)',
                                    'N° de ocurrencias', con_leyenda=False, rango_x=[0, x_max_rango]
                                )
                                fig_conteo.update_layout(bargap=0.04)
                                st.plotly_chart(fig_conteo, width='stretch')

                            if sin_dist_entrada:
                                ayuda_inline("Esta actividad/demora no tiene distribución de entrada asociada (ej. demoras por restricciones o fallas con demora manual): solo se muestran los resultados simulados.", "Sin distribución de entrada")
                            st.caption(f"Túnel: {tunel_seleccionado} — {titulo_mc}", help="Izquierda: densidad de probabilidad; las líneas finas de abajo son cada resultado simulado, comparado con la distribución de entrada (si existe). Derecha: conteo real de veces que la simulación cayó en cada intervalo de tiempo.")

            if not tunel_seleccionado:
                for _tab_vacia in (tab_mf, tab_traza, tab_export):
                    with _tab_vacia:
                        st.info("No hay túneles disponibles para mostrar estos resultados.")
            elif not st.session_state.get('mantenciones_programadas') and not st.session_state.get('fallas_equipos'):
                with tab_mf:
                    st.info("No se configuraron mantenciones programadas ni fallas de equipos en esta simulación.")

            if tunel_seleccionado:  # Menús 19, 20, 21 y 22 (misma condición que la Carta Gantt)

                # ----------------------------------------------------
                # Resumen de Mantenciones Programadas Ejecutadas
                # ----------------------------------------------------

                # ======================================================================
                # #####Menu 19 — Resumen de Mantenciones Programadas Ejecutadas#####
                # ======================================================================
                if st.session_state.get('mantenciones_programadas'):
                    with tab_mf, seccion_menu('19', 'Resultados de Mantenciones Programadas'):
                        df_mant_res = st.session_state.df_traza_eventos_cache
                        df_mant_res = df_mant_res[
                            df_mant_res['type'].isin(['maintenance', 'maintenance_failed'])
                        ] if not df_mant_res.empty and 'type' in df_mant_res.columns else pd.DataFrame()

                        if df_mant_res.empty:
                            st.caption("Ninguna mantención programada cayó dentro del horizonte simulado.", help="Revise que el inicio configurado no supere el tiempo límite de simulación.")
                        else:
                            resumen_mant = (
                                df_mant_res.groupby(['Simulation_ID', 'Recurso'])
                                .agg(
                                    Intentos=('type', 'count'),
                                    Intentos_Fallidos=('type', lambda s: (s == 'maintenance_failed').sum()),
                                    Tiempo_Total_Indisponible_h=('Duration', 'sum'),
                                )
                                .reset_index()
                                .rename(columns={
                                    'Simulation_ID': 'Simulación', 'Recurso': 'Equipo',
                                    'Intentos': 'N° Intentos', 'Intentos_Fallidos': 'Intentos Fallidos',
                                    'Tiempo_Total_Indisponible_h': 'Tiempo Total Fuera de Servicio (h)'
                                })
                            )
                            resumen_mant['Tiempo Total Fuera de Servicio (h)'] = resumen_mant['Tiempo Total Fuera de Servicio (h)'].round(2)
                            mostrar_tabla(resumen_mant, width='stretch', hide_index=True)
                            ayuda_inline("Intento Fallido: mantención que no tuvo éxito (según la Probabilidad de Éxito configurada) y encadenó un reintento inmediato. Tiempo Total Fuera de Servicio: suma todos los intentos, exitosos y fallidos, de esa simulación para ese equipo.", "Cómo leer la tabla")

                            # ---- Línea de tiempo de las ventanas de mantención (una fila por equipo) ----
                            sims_mant = sorted(df_mant_res['Simulation_ID'].unique())
                            sim_mant_sel = st.selectbox(
                                "Simulación a visualizar (mantenciones por equipo)", sims_mant, key="sim_mant_14b"
                            )
                            df_mant_sim = df_mant_res[df_mant_res['Simulation_ID'] == sim_mant_sel].sort_values('Start')
                            fig_mant = go.Figure()
                            for tipo_m, nombre_m, color_m in (
                                ('maintenance', 'Mantención (éxito)', '#00838F'),
                                ('maintenance_failed', 'Mantención (falló, reintento)', '#455A64')
                            ):
                                d_m = df_mant_sim[df_mant_sim['type'] == tipo_m]
                                if d_m.empty:
                                    continue
                                fig_mant.add_trace(go.Bar(
                                    x=d_m['Duration'], y=d_m['Recurso'], base=d_m['Start'], orientation='h',
                                    marker_color=color_m, name=nombre_m,
                                    text=d_m['Duration'].apply(lambda v: f"{fmt2(v)} h"), textposition='auto',
                                    hovertext=[f"<b>{a}</b><br>Inicio: {fmt2(i)} h<br>Fin: {fmt2(f)} h<br>Duración: {fmt2(d)} h"
                                               for a, i, f, d in zip(d_m['Actividad'], d_m['Start'], d_m['Finish'], d_m['Duration'])],
                                    hoverinfo='text'
                                ))
                            fig_mant.update_layout(
                                title=f"Ventanas de mantención por equipo — Simulación {sim_mant_sel}",
                                xaxis_title="Tiempo (h)", yaxis_title="", barmode='overlay',
                                height=max(220, 60 * df_mant_sim['Recurso'].nunique() + 140)
                            )
                            st.plotly_chart(fig_mant, width='stretch', key="grafico_mantenciones_14b")

                            # ---- Impacto en los túneles (esperas/pausas provocadas) ----
                            df_imp_mant = st.session_state.df_traza_eventos_cache
                            df_imp_mant = df_imp_mant[df_imp_mant['type'] == 'delay_maintenance']
                            if df_imp_mant.empty:
                                st.caption("Ningún túnel tuvo que esperar por estas mantenciones en las simulaciones corridas.")
                            else:
                                n_sims_imp = max(int(df_imp_mant['Simulation_ID'].nunique()), 1)
                                imp = (
                                    df_imp_mant.groupby(['Resource_Origin_Tunnel', 'Recurso'])
                                    .agg(Eventos=('Duration', 'count'), Total=('Duration', 'sum'),
                                         Simulaciones=('Simulation_ID', 'nunique'))
                                    .reset_index()
                                )
                                imp['Espera Promedio por Simulación (h)'] = (imp['Total'] / n_sims_imp).round(2)
                                imp = imp.rename(columns={
                                    'Resource_Origin_Tunnel': 'Túnel afectado', 'Recurso': 'Equipo en mantención',
                                    'Eventos': 'N° Esperas/Pausas (todas las sims)', 'Simulaciones': 'Sims. con efecto'
                                }).drop(columns='Total')
                                st.markdown("**Impacto en los túneles (tiempo que esperaron o pausaron por mantención):**")
                                mostrar_tabla(imp, width='stretch', hide_index=True)



                # ----------------------------------------------------
                # Resumen de Fallas de Equipos Ocurridas
                # ----------------------------------------------------

                # ======================================================================
                # #####Menu 20 — Resumen de Fallas de Equipos Ocurridas#####
                # ======================================================================
                if st.session_state.get('fallas_equipos'):
                    with tab_mf, seccion_menu('20', 'Resultados de Fallas de Equipos'):
                        df_traza_14c = st.session_state.df_traza_eventos_cache
                        escenarios_14c = st.session_state.get('fallas_equipos', [])

                        # Misma regla que el motor: si un túnel está en varios escenarios, manda el ÚLTIMO.
                        config_falla_por_tunel = {}
                        for esc_14c in escenarios_14c:
                            for fr_14c in esc_14c.get('frentes_aplicables', []):
                                config_falla_por_tunel[fr_14c] = esc_14c
                        if not df_traza_14c.empty and 'Resource_Origin_Tunnel' in df_traza_14c.columns:
                            tuneles_14c = [t for t in sorted(df_traza_14c['Resource_Origin_Tunnel'].dropna().unique())
                                           if t in config_falla_por_tunel]
                        else:
                            tuneles_14c = []

                        if not tuneles_14c:
                            st.info("Ninguno de los túneles simulados tiene un escenario de falla configurado (Configuración de Fallas de Equipos).")
                        else:
                            c14_t, c14_s = st.columns([1, 1])
                            idx_def_14c = tuneles_14c.index(tunel_seleccionado) if tunel_seleccionado in tuneles_14c else 0
                            tunel_14c = c14_t.selectbox("Túnel", tuneles_14c, index=idx_def_14c, key="falla_res_tunel")
                            esc_t = config_falla_por_tunel[tunel_14c]
                            cfg_recursos = esc_t.get('fallas_por_recurso', {})

                            # ---- 1) Lo CONFIGURADO para este túnel ----
                            filas_cfg = []
                            for rec_cfg, cfg in cfg_recursos.items():
                                params_cfg = cfg.get('params_distribucion')
                                lo_cfg = hi_cfg = None
                                if cfg.get('modo_demora') == 'Distribución estadística':
                                    if isinstance(params_cfg, dict):
                                        lo_cfg, hi_cfg = params_cfg.get('_trunc_lo'), params_cfg.get('_trunc_hi')
                                        base_txt = ", ".join(f"{k}={v:g}" for k, v in params_cfg.items()
                                                             if not str(k).startswith('_') and isinstance(v, (int, float)))
                                    else:
                                        base_txt = f"{params_cfg}"
                                    dem_txt = f"{cfg.get('distribucion')} ({base_txt})"
                                    rango_txt = f"[{lo_cfg:.2f}, {hi_cfg:.2f}] h" if lo_cfg is not None else "sin rango (distribución completa)"
                                else:
                                    dem_txt = f"{float(cfg.get('demora') or 0):g} h fija"
                                    rango_txt = "—"
                                filas_cfg.append({
                                    'Equipo': rec_cfg, 'Prob. de falla por ciclo (%)': cfg.get('probabilidad'),
                                    'Demora configurada': dem_txt, 'Rango permitido': rango_txt,
                                })
                            st.markdown(f"#### Configuración de fallas del túnel {tunel_14c} (escenario '{esc_t.get('nombre')}')")
                            if filas_cfg:
                                mostrar_tabla(pd.DataFrame(filas_cfg), width='stretch', hide_index=True)
                            else:
                                st.caption("Este escenario no tiene recursos con probabilidad de falla > 0.")

                            # ---- 2) Lo que SORTEÓ la simulación en este túnel ----
                            es_falla_14c = df_traza_14c['type'] == 'delay_equipment_failure'
                            df_f_t = df_traza_14c[es_falla_14c & (df_traza_14c['Resource_Origin_Tunnel'] == tunel_14c)].copy()
                            sims_14c = sorted(df_traza_14c['Simulation_ID'].unique())
                            OPCION_TODAS_14C = "Todas (cada una por separado)"
                            sim_sel_14c = c14_s.selectbox("Simulación", [OPCION_TODAS_14C] + [int(x) for x in sims_14c],
                                                          key="falla_res_sim")

                            # Avances (ciclos) iniciados por simulación en este túnel
                            df_ciclos_t = df_traza_14c[(df_traza_14c['Resource_Origin_Tunnel'] == tunel_14c)
                                                       & (df_traza_14c['type'] != 'travel')
                                                       & (df_traza_14c['Ciclo_Frente'] >= 1)]
                            ciclos_por_sim = df_ciclos_t.groupby('Simulation_ID')['Ciclo_Frente'].nunique()

                            if df_f_t.empty:
                                st.info(f"En el túnel {tunel_14c} no ocurrió ninguna falla en las simulaciones "
                                        "(con probabilidades bajas puede que no se hayan gatillado).")
                            else:
                                # Rango permitido por equipo (para validar lo sorteado)
                                def _rango_equipo(r):
                                    cfg = cfg_recursos.get(r)
                                    p = cfg.get('params_distribucion') if cfg else None
                                    if cfg and cfg.get('modo_demora') == 'Distribución estadística' and isinstance(p, dict):
                                        return p.get('_trunc_lo'), p.get('_trunc_hi')
                                    return None, None
                                rangos = df_f_t['Recurso'].map(_rango_equipo)
                                df_f_t['_lo'] = [x[0] for x in rangos]
                                df_f_t['_hi'] = [x[1] for x in rangos]
                                tol = 0.006  # 'Duration' se guarda redondeada a 2 decimales
                                con_rango = df_f_t['_lo'].notna()
                                df_f_t['_ok'] = np.where(
                                    con_rango,
                                    (df_f_t['Duration'] >= df_f_t['_lo'] - tol) & (df_f_t['Duration'] <= df_f_t['_hi'] + tol),
                                    True)

                                df_vista = df_f_t if sim_sel_14c == OPCION_TODAS_14C else df_f_t[df_f_t['Simulation_ID'] == sim_sel_14c]

                                # Resumen POR SIMULACIÓN (nunca se suman simulaciones entre sí)
                                res_sim = (
                                    df_vista.groupby(['Simulation_ID', 'Recurso'])
                                    .agg(N=('type', 'count'), Total=('Duration', 'sum'),
                                         Min=('Duration', 'min'), Prom=('Duration', 'mean'), Max=('Duration', 'max'))
                                    .reset_index()
                                )
                                res_sim['Avances iniciados'] = res_sim['Simulation_ID'].map(ciclos_por_sim).fillna(0).astype(int)
                                st.markdown(f"#### Fallas sorteadas en {tunel_14c}, por simulación")
                                mostrar_tabla(pd.DataFrame({
                                    'Simulación': res_sim['Simulation_ID'], 'Equipo': res_sim['Recurso'],
                                    'Avances iniciados': res_sim['Avances iniciados'],
                                    'N° de fallas': res_sim['N'],
                                    'Demora mín (h)': res_sim['Min'].round(2), 'Demora prom (h)': res_sim['Prom'].round(2),
                                    'Demora máx (h)': res_sim['Max'].round(2), 'Demora total (h)': res_sim['Total'].round(2),
                                }), width='stretch', hide_index=True)

                                n_fuera = int((~df_vista['_ok']).sum())
                                if con_rango.any():
                                    if n_fuera == 0:
                                        st.success(f"Las {int(df_vista['_lo'].notna().sum())} demoras sorteadas con rango "
                                                   "configurado están dentro del rango permitido.")
                                    else:
                                        st.error(f"{n_fuera} demora(s) sorteadas quedaron FUERA del rango configurado "
                                                 "(ver la columna 'Dentro del rango' abajo).")

                                # Detalle de cada falla sorteada
                                detalle_f = df_vista.sort_values(['Simulation_ID', 'Start'])
                                tabla_det = pd.DataFrame({
                                    'Simulación': detalle_f['Simulation_ID'],
                                    'Avance': detalle_f['Ciclo_Frente'],
                                    'Equipo': detalle_f['Recurso'],
                                    'Inicio (h)': detalle_f['Start'],
                                    'Demora sorteada (h)': detalle_f['Duration'].round(2),
                                    'Rango permitido': [f"[{lo:.2f}, {hi:.2f}]" if pd.notna(lo) else "—"
                                                        for lo, hi in zip(detalle_f['_lo'], detalle_f['_hi'])],
                                    'Dentro del rango': np.where(detalle_f['_lo'].isna(), "—",
                                                                 np.where(detalle_f['_ok'], "Sí", "No")),
                                })
                                with st.expander("Detalle de cada falla sorteada", expanded=True):
                                    mostrar_tabla(tabla_det, width='stretch', hide_index=True)

                                # Gráfico: demora sorteada por falla vs. rango permitido
                                fig_f = go.Figure()
                                etiquetas_x = [f"Sim {int(s)} · Av {int(a)}" for s, a in
                                               zip(detalle_f['Simulation_ID'], detalle_f['Ciclo_Frente'])]
                                fig_f.add_trace(go.Scatter(
                                    x=etiquetas_x, y=detalle_f['Duration'], mode='markers',
                                    marker=dict(size=9, color=np.where(detalle_f['_ok'], '#1f77b4', '#d62728')),
                                    text=detalle_f['Recurso'], name='Demora sorteada',
                                    hovertemplate="%{x}<br>%{text}<br>Demora: %{y:.2f} h<extra></extra>"))
                                lo_g, hi_g = detalle_f['_lo'].dropna(), detalle_f['_hi'].dropna()
                                if not lo_g.empty and lo_g.nunique() == 1:
                                    fig_f.add_hrect(y0=float(lo_g.iloc[0]), y1=float(hi_g.iloc[0]),
                                                    fillcolor='rgba(44,160,44,0.15)', line_width=0,
                                                    annotation_text="Rango permitido", annotation_position="top left")
                                fig_f.update_layout(
                                    title=f"Demora de cada falla sorteada — {tunel_14c}",
                                    yaxis_title="Demora (h)", xaxis_title="Simulación · Avance",
                                    height=380, showlegend=False)
                                st.plotly_chart(fig_f, width='stretch')

                            ayuda_inline("Cada fila es una falla real sorteada por la simulación (una por equipo como máximo en cada avance, con la probabilidad configurada). Las simulaciones no se suman entre sí. Avances iniciados = avances de ese túnel que la simulación llegó a comenzar.", "Cómo leer la tabla")



                # ----------------------------------------------------
                # Traza de Eventos Detallada
                # ----------------------------------------------------

                # ======================================================================
                # #####Menu 21 — Traza de Eventos Detallada#####
                # ======================================================================
                with tab_traza, seccion_menu('21', 'Traza de Eventos Detallada'):

                    # ===== NUEVA PESTAÑA: VISTA POR RECURSO (COMPLETA) =====
                    tab1, tab2 = st.tabs(["Vista por Túnel (Filtrada)", "Vista por Recurso (Completa)"])
                
                    with tab1:
                        # ===== CÓDIGO ORIGINAL (VISTA FILTRADA POR TÚNEL) =====
                        col_sim, col_filtro = st.columns([1, 1])

                        with col_sim:
                            simulacion_detallada = st.selectbox(
                                "Seleccione el ID de la Simulación para la traza", 
                                df_traza_completa['Simulation_ID'].unique(), 
                                index=0,
                                key="sim_filtrada"
                            )

                        # Eventos del tunel en la simulacion elegida, ya con su AVANCE y ordenados por
                        # avance (se calcula antes del filtro para poder listar los avances disponibles).
                        df_traza_sim = pd.DataFrame()
                        avances_traza_disp = []
                        if simulacion_detallada:
                            df_traza_sim = preparar_traza_tunel_por_avance(
                                df_traza_completa, simulacion_detallada, tunel_seleccionado
                            )
                            avances_traza_disp = sorted(int(a) for a in df_traza_sim['Avance'].unique() if a >= 1)

                        avance_traza_sel = None
                        with col_filtro:
                            # Filtro por avance del tunel (Avance 1, Avance 2, ...)
                            ver_todos_avances = st.checkbox("Ver todos los avances", value=True, key="avances_filtrada")

                            if not ver_todos_avances and avances_traza_disp:
                                avance_traza_sel = st.selectbox(
                                    "Seleccione un avance específico",
                                    avances_traza_disp,
                                    format_func=lambda a: f"Avance {a}",
                                    key="avance_filtrada"
                                )

                        if simulacion_detallada:
                            # Filtrar por avance si no se quieren ver todos
                            if avance_traza_sel is not None:
                                df_traza_sim = df_traza_sim[df_traza_sim['Avance'] == avance_traza_sel].reset_index(drop=True)

                            # Ya viene ordenada por avance y, dentro de cada avance, cronologicamente.
                            # Agregar número de fila
                            df_traza_sim['Secuencia_Temporal'] = range(1, len(df_traza_sim) + 1)

                            df_output = df_traza_sim[[
                                'Secuencia_Temporal', 'Avance', 'Sec.', 'Actividad', 'Recurso', 'type', 'Start', 'Finish', 'Duration',
                                'Start_X_Front', 'Start_Y_Front', 'End_X_Front', 'End_Y_Front',
                                'Cycle_Advance_Length_m', 'Resource_Origin_Tunnel', 'Resource_Destination_Tunnel',
                                'Travel_Time_Actual', 'Travel_Speed_Used_m_h', 'Travel_Distance_m'
                            ]].rename(columns={
                                'Secuencia_Temporal': '#',
                                'type': 'Tipo Evento', 'Start': 'Inicio (h)', 'Finish': 'Fin (h)',
                                'Duration': 'Duración (h)', 'Start_X_Front': 'Inicio X', 'Start_Y_Front': 'Inicio Y',
                                'End_X_Front': 'Fin X', 'End_Y_Front': 'Fin Y', 'Cycle_Advance_Length_m': 'Avance Ciclo (m)',
                                'Resource_Origin_Tunnel': 'Túnel Origen', 'Resource_Destination_Tunnel': 'Túnel Destino',
                                'Travel_Time_Actual': 'Tiempo Viaje (h)', 'Travel_Speed_Used_m_h': 'Velocidad Viaje (km/h)',
                                'Travel_Distance_m': 'Distancia Viaje (m)'
                            })

                            # La traza guarda la velocidad en m/h; en pantalla se muestra en km/h.
                            df_output['Velocidad Viaje (km/h)'] = df_output['Velocidad Viaje (km/h)'] / 1000.0

                            df_output['Tipo Evento'] = df_output['Tipo Evento'].map({
                                'travel': 'VIAJE', 
                                'activity': 'ACTIVIDAD', 
                                'delay_fh_oc': 'Demora (FH/OC)', 
                                'delay_res': 'Demora (R-Turno)',
                                'delay_bloqueo_total': 'Bloqueo Total de Túnel',
                                'delay_equipment_failure': 'Demora (Falla Equipo)',
                                'delay_fleet_change': 'Demora (Cambio de Recurso)',
                                'delay_maintenance': 'Demora (Mantención Programada)',
                                'delay_shift_change': 'Demora (Cambio de Turno)',
                                'maintenance': 'Mantención Programada',
                                'maintenance_failed': 'Mantención Programada (falló, reintento)'
                            }).fillna(df_output['Tipo Evento'])
                        
                            # Columna visual para identificar eventos del túnel seleccionado
                            df_output['Ubicación'] = df_output.apply(
                                lambda row: f'{tunel_seleccionado}' if row['Túnel Destino'] == tunel_seleccionado 
                                else f'{row["Túnel Destino"]}' if row['Túnel Destino'] != 'N/A'
                                else 'Demora',
                                axis=1
                            )

                            # Estilo condicional mejorado
                            def highlight_events(row):
                                if '' in str(row['Ubicación']):
                                    return ['background-color: #d4edda'] * len(row)  # Verde: en túnel seleccionado
                                elif row['Tipo Evento'] == 'VIAJE':
                                    return ['background-color: #fff3cd'] * len(row)  # Amarillo: viajando
                                elif 'Demora' in str(row['Tipo Evento']):
                                    return ['background-color: #f8d7da'] * len(row)  # Rojo claro: demora
                                else:
                                    return ['background-color: #e7f3ff'] * len(row)  # Azul claro: otro túnel
                        
                            st.markdown(f"#### Eventos en **{tunel_seleccionado}** (por Avance, en Orden Cronológico)")
                            st.caption(f"{len(df_output)} eventos de este túnel, agrupados por avance", help="Dentro de cada avance: demoras, viajes de llegada, actividades (en orden de secuencia) y viajes de salida. Los viajes de llegada van en el avance del túnel al que llegan; los de salida, en el avance que estaba en curso cuando el recurso partió.")
                        
                            # Nota: el Styler de pandas usa por defecto 6 decimales al mostrar
                            # columnas numericas, aunque el valor ya este redondeado a 2. Se fuerza
                            # el formato con fmt2 (maximo 2 decimales, con cifras significativas
                            # para valores muy pequeños) para que se vea limpio.
                            cols_enteras_output = ['#', 'Avance']
                            cols_numericas_output = [
                                c for c in df_output.select_dtypes(include=[np.number]).columns.tolist()
                                if c not in cols_enteras_output
                            ]
                            mostrar_tabla(
                                df_output.style.apply(highlight_events, axis=1)
                                .format(fmt2, subset=cols_numericas_output)
                                .format(lambda v: '—' if int(v) == 0 else f"{int(v)}", subset=['Avance'])
                                .format("{:.0f}", subset=['#']),
                                width='stretch', 
                                hide_index=True,
                                height=600
                            )
                        
                            # Mantenciones programadas de EQUIPOS (no pertenecen a un túnel): se listan siempre,
                            # para que el usuario vea cuándo estuvo cada equipo fuera de servicio.
                            df_mant_tab1 = df_traza_completa[
                                (df_traza_completa['Simulation_ID'] == simulacion_detallada)
                                & (df_traza_completa['type'].isin(['maintenance', 'maintenance_failed']))
                            ].sort_values('Start')
                            if not df_mant_tab1.empty:
                                with st.expander(f"Mantenciones programadas de equipos en la Simulación {simulacion_detallada} "
                                                 f"({len(df_mant_tab1)} evento(s))", expanded=True):
                                    df_mant_tab1_v = pd.DataFrame({
                                        'Equipo': df_mant_tab1['Recurso'].values,
                                        'Mantención': df_mant_tab1['Detalle_Mantencion'].values
                                        if 'Detalle_Mantencion' in df_mant_tab1.columns else df_mant_tab1['Actividad'].values,
                                        'Resultado': np.where(df_mant_tab1['type'].values == 'maintenance', 'Éxito', 'Falló (reintento)'),
                                        'Inicio (h)': df_mant_tab1['Start'].values,
                                        'Fin (h)': df_mant_tab1['Finish'].values,
                                        'Duración (h)': df_mant_tab1['Duration'].values,
                                    })
                                    mostrar_tabla(df_mant_tab1_v.style.format(fmt2, subset=['Inicio (h)', 'Fin (h)', 'Duración (h)']),
                                                  width='stretch', hide_index=True)
                                    ayuda_inline("Las esperas/pausas que estas mantenciones provocaron en este túnel aparecen arriba, dentro de cada avance, como Demora (Mantención Programada).", "Dónde ver las demoras")

                            ayuda_inline(f"Esta vista muestra solo eventos donde {tunel_seleccionado} es origen o destino. Para ver el recorrido completo de recursos entre TODOS los túneles, use la pestaña Vista por Recurso.", "Alcance de la vista")
                
                    # ===== NUEVA TAB: VISTA COMPLETA POR RECURSO =====
                    #

                    # ===== NUEVA TAB: VISTA COMPLETA POR RECURSO =====
                    with tab2:
                        st.markdown("#### Seguimiento Completo de Recursos entre Tuneles")
                    
                        col_sim2, col_rec = st.columns([1, 1])
                    
                        with col_sim2:
                            simulacion_recurso = st.selectbox(
                                "Seleccione el ID de la Simulacion", 
                                df_traza_completa['Simulation_ID'].unique(), 
                                index=0,
                                key="sim_recurso"
                            )
                    
                        with col_rec:
                            # Obtener recursos únicos (sin N/A)
                            recursos_disponibles = sorted(
                                df_traza_completa[
                                    (df_traza_completa['Recurso'].notna()) & 
                                    (df_traza_completa['Recurso'] != 'N/A')
                                ]['Recurso'].unique()
                            )
                        
                            if recursos_disponibles:
                                recurso_seleccionado = st.selectbox(
                                    "Seleccione el Recurso a seguir",
                                    recursos_disponibles,
                                    key="recurso_select"
                                )
                            else:
                                st.warning("No hay recursos con identificador en la traza")
                                recurso_seleccionado = None
                    
                        if simulacion_recurso and recurso_seleccionado:
                            # Filtrar SOLO por simulación y recurso (SIN filtro de túnel)
                            df_recurso_completo = df_traza_completa[
                                (df_traza_completa['Simulation_ID'] == simulacion_recurso) &
                                (df_traza_completa['Recurso'] == recurso_seleccionado)
                            ].copy()
                        
                            # Ordenar cronológicamente
                            df_recurso_completo = df_recurso_completo.sort_values(by='Start').reset_index(drop=True)
                            df_recurso_completo['Secuencia'] = range(1, len(df_recurso_completo) + 1)
                        
                            # Preparar tabla de salida
                            # 'Avance' = N° de avance del túnel donde trabaja el recurso (1, 2, 3...), que es
                            # 'Ciclo_Frente'. 'Cycle' es un contador global de la simulación (no sirve de referencia).
                            # Los eventos sin avance (mantenciones, cambios de flota) quedan con '—'.
                            if 'Ciclo_Frente' in df_recurso_completo.columns:
                                _av = pd.to_numeric(df_recurso_completo['Ciclo_Frente'], errors='coerce')
                            else:
                                _av = pd.Series(np.nan, index=df_recurso_completo.index)
                            df_recurso_completo['Avance'] = [
                                int(v) if pd.notna(v) and v >= 1 else "—" for v in _av
                            ]
                            df_recurso_completo['Secuencia'] = df_recurso_completo['Secuencia'].astype(int)
                            df_output_recurso = df_recurso_completo[[
                                'Secuencia', 'Avance', 'Actividad', 'type', 'Start', 'Finish', 'Duration',
                                'Resource_Origin_Tunnel', 'Resource_Destination_Tunnel',
                                'Travel_Distance_m', 'Travel_Speed_Used_m_h'
                            ]].rename(columns={
                                'Secuencia': '#',
                                'type': 'Tipo',
                                'Start': 'Inicio (h)',
                                'Finish': 'Fin (h)',
                                'Duration': 'Duracion (h)',
                                'Resource_Origin_Tunnel': 'Tunel Origen',
                                'Resource_Destination_Tunnel': 'Tunel Destino',
                                'Travel_Distance_m': 'Distancia (m)',
                                'Travel_Speed_Used_m_h': 'Velocidad (km/h)'
                            })
                            # La traza guarda la velocidad en m/h; en pantalla se muestra en km/h.
                            df_output_recurso['Velocidad (km/h)'] = df_output_recurso['Velocidad (km/h)'] / 1000.0
                        
                            # Traducir tipos de evento
                            df_output_recurso['Tipo'] = df_output_recurso['Tipo'].map({
                                'travel': 'VIAJE',
                                'activity': 'ACTIVIDAD',
                                'delay_fh_oc': 'Demora FH/OC',
                                'delay_res': 'Demora Turno',
                                'delay_bloqueo_total': 'Bloqueo Total de Túnel',
                                'delay_equipment_failure': 'Demora Falla',
                                'delay_fleet_change': 'Demora Cambio Recurso',
                                'delay_maintenance': 'Demora Mantención Programada',
                                'delay_shift_change': 'Demora Cambio de Turno',
                                'maintenance': 'Mantención Programada',
                                'maintenance_failed': 'Mantención Programada (falló)'
                            }).fillna(df_output_recurso['Tipo'])
                        
                            # Crear columna resumen del movimiento
                            df_output_recurso['Movimiento'] = df_output_recurso.apply(
                                lambda row: f"{row['Tunel Origen']} -> {row['Tunel Destino']}" 
                                if row['Tipo'] == 'VIAJE' 
                                else ("Equipo fuera de servicio (mantención)" if 'Mantención Programada' == str(row['Tipo']) or 'falló' in str(row['Tipo'])
                                      else (f"Esperando mantención (frente {row['Tunel Origen']})" if 'Mantención' in str(row['Tipo'])
                                            else f"Trabajando en {row['Tunel Destino']}")),
                                axis=1
                            )
                        
                            # Estilo condicional por tipo de evento
                            def highlight_recurso_events(row):
                                if 'VIAJE' in str(row['Tipo']):
                                    return ['background-color: #fff3cd'] * len(row)
                                elif 'ACTIVIDAD' in str(row['Tipo']):
                                    return ['background-color: #d1ecf1'] * len(row)
                                elif 'Mantención' in str(row['Tipo']):
                                    return ['background-color: #cdeff2'] * len(row)
                                elif 'Demora' in str(row['Tipo']):
                                    return ['background-color: #f8d7da'] * len(row)
                                else:
                                    return [''] * len(row)
                        
                            st.markdown(f"##### Recorrido de {recurso_seleccionado} (Simulacion #{simulacion_recurso})")
                            st.caption(f"Total de eventos: {len(df_output_recurso)} | Tuneles visitados: {df_output_recurso['Tunel Destino'].nunique()}")
                        
                            # Mostrar tabla con estilo
                            df_output_recurso_vista = df_output_recurso[[
                                '#', 'Avance', 'Tipo', 'Actividad', 'Movimiento', 
                                'Inicio (h)', 'Fin (h)', 'Duracion (h)', 
                                'Distancia (m)', 'Velocidad (km/h)'
                            ]]
                            # '#' y 'Avance' son enteros (1, 2, 3...): no pasan por el formato de decimales.
                            cols_numericas_recurso = [c for c in df_output_recurso_vista.select_dtypes(include=[np.number]).columns
                                                      if c not in ('#', 'Avance')]
                            mostrar_tabla(
                                df_output_recurso_vista.style.apply(highlight_recurso_events, axis=1).format(fmt2, subset=cols_numericas_recurso),
                                width='stretch',
                                hide_index=True,
                                height=600
                            )
                        
                            ayuda_inline("Verificación de liberación inmediata: este recurso trabajó en múltiples túneles y cada actividad está seguida inmediatamente por un viaje o una nueva actividad, sin tiempos muertos.", "Verificación de liberación")

                # ===== SECCION DE EXPORTACION A EXCEL =====


                # ======================================================================
                # #####Menu 22 — Exportación de Resultados a Excel#####
                # ======================================================================
                with tab_export, seccion_menu('22', 'Exportación de Resultados a Excel'):

                    ruta_definida = st.session_state.get('ruta_salida', '').strip()

                    col_exp1, col_exp2 = st.columns(2)
                    with col_exp1:
                        guardar_en_ruta_btn = st.button(
                            "Guardar en Ruta Definida (6 archivos + reporte)",
                            type="primary",
                            width='stretch',
                            disabled=not bool(ruta_definida),
                            help="Guarda los 6 archivos formato David Montenegro + reporte combinado en la ruta definida en «Ejecutar Simulación»."
                        )
                    with col_exp2:
                        descargar_btn = st.button(
                            "Generar Reporte Descargable (1 archivo combinado)",
                            type="primary",
                            width='stretch'
                        )

                    if guardar_en_ruta_btn and ruta_definida:
                        with st.spinner("Guardando archivos en ruta..."):
                            try:
                                archivos = guardar_outputs_en_ruta(
                                    output_path=ruta_definida,
                                    traza_eventos=st.session_state.traza_eventos,
                                    resultados=st.session_state.resultados,
                                    frentes_info=st.session_state.frentes_info,
                                    df_frentes_nivel=st.session_state.get('df_frentes_nivel_simuladas', frentes_nivel),
                                    df_actividades=st.session_state.get('df_actividades_simuladas', actividades_nivel),
                                    recursos_config=st.session_state.recursos_config,
                                    numero_turnos=st.session_state.get('numero_turnos_simulado', st.session_state.get('numero_turnos', 62)),
                                    duracion_turno=st.session_state.get('duracion_turno_simulada', st.session_state.get('duracion_turno_base', 8.0)),
                                    n_simulaciones=st.session_state.get('n_simulaciones_ejecutadas', int(st.session_state.get('n_simulaciones', 1000))),
                                    nivel_seleccionado=st.session_state.get('nivel_simulado', nivel_seleccionado)
                                )
                                st.success(f"{len(archivos)} archivos guardados en `{ruta_definida}`:")
                                for a in archivos:
                                    st.text(f"  • {os.path.basename(a)}")
                            except Exception as e:
                                st.error(f"Error al guardar: {e}")

                    if not ruta_definida:
                        st.info("Streamlit Cloud: use «Generar Reporte Descargable» para bajar el Excel a su equipo."
                                if ES_NUBE else
                                "Define la ruta en «Ejecutar Simulación» para habilitar el guardado directo.")

                    if descargar_btn or (not guardar_en_ruta_btn and not descargar_btn):
                        # Solo mostrar la sección de descarga si se presionó el botón de descarga
                        pass

                    if descargar_btn:
                        with st.spinner("Generando reporte Excel..."):
                            output = BytesIO()
                            tuneles_a_exportar = frentes_seleccionados
                        
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                # HOJA 1: Resumen General
                                resumen_general = []
                                resumen_general.append(['REPORTE DE SIMULACION DE TUNELES', ''])
                                resumen_general.append(['Nivel Seleccionado', nivel_seleccionado])
                                resumen_general.append(['Sistema de Turnos', st.session_state.get('sistema_turnos', '')])
                                resumen_general.append(['Numero de Turnos', st.session_state.get('numero_turnos', '')])
                                resumen_general.append(['Tiempo Limite (horas)', st.session_state.get('tiempo_limite', 0)])
                                resumen_general.append(['Metros por Ciclo por Defecto', st.session_state.get('metros_avance', 3.5)])
                                resumen_general.append(['Numero de Simulaciones', st.session_state.get('n_simulaciones', 1000)])
                                resumen_general.append(['Ruta Critica', ", ".join(st.session_state.get('ruta_critica', []))])
                                resumen_general.append(['Restricciones Fisicas (FH, OC, R)', 'Activadas' if st.session_state.get('restricciones_fisicas_activas', True) else 'Desactivadas'])
                                resumen_general.append(['Bloqueo Total de Tuneles', 'Activado' if st.session_state.get('bloqueo_total_activo', False) else 'Desactivado'])
                                resumen_general.append(['Radio FH/OC (m)', radio_restriccion])
                                resumen_general.append(['Demora por Restriccion (h)', demora_horas])
                                resumen_general.append(['', ''])
                                resumen_general.append(['TUNELES SIMULADOS', 'DISTANCIA TOTAL (m)', 'SECCION', 'METROS POR CICLO'])
                                for frente, info in st.session_state.frentes_info.items():
                                    resumen_general.append([
                                        frente,
                                        round(info['distancia'], 2),
                                        info.get('seccion', ''),
                                        info.get('metros_por_ciclo', st.session_state.get('metros_avance', 3.5))
                                    ])
                            
                                df_resumen = pd.DataFrame(resumen_general)
                                df_resumen.to_excel(writer, sheet_name='Resumen General', index=False, header=False)

                                pd.DataFrame(
                                    [{'Seccion': k, 'Metros por Ciclo': v} for k, v in st.session_state.get('metros_por_seccion', {}).items()]
                                ).to_excel(writer, sheet_name='Metros por Seccion', index=False)
                                pd.DataFrame(
                                    [{'Tipo': k, 'Radio (m)': v} for k, v in st.session_state.get('radios_por_tipo', {}).items()]
                                ).to_excel(writer, sheet_name='Radios Restricciones', index=False)
                                if st.session_state.get('restricciones_geologicas'):
                                    pd.DataFrame(st.session_state.restricciones_geologicas).to_excel(writer, sheet_name='Restricciones Geo', index=False)
                                if st.session_state.get('mwd_config_usada') and st.session_state.get('mwd_log_geologia') is not None:
                                    _u_mwd = st.session_state.mwd_config_usada
                                    pd.DataFrame([
                                        {'Parametro': 'Semiancho (puntos percentiles)', 'Valor': _u_mwd['semiancho']},
                                        {'Parametro': 'Reglas por actividad', 'Valor': str(_u_mwd['reglas'])},
                                    ] + [{'Parametro': f'Tunel {t}', 'Valor': str(a)} for t, a in _u_mwd['asignaciones'].items()]
                                    ).to_excel(writer, sheet_name='Config Geologia MWD', index=False)
                                    st.session_state.mwd_log_geologia.to_excel(writer, sheet_name='Geologia MWD por Bloque', index=False)
                                    if st.session_state.get('mwd_catalogo_usado') is not None:
                                        st.session_state.mwd_catalogo_usado.to_excel(writer, sheet_name='Catalogo MWD', index=False)
                                if st.session_state.get('fallas_equipos'):
                                    # Version "aplanada" (una fila por recurso dentro de cada escenario)
                                    # mas legible que un volcado directo de la estructura anidada.
                                    filas_falla_export = []
                                    for esc in st.session_state.fallas_equipos:
                                        for recurso_f, v_f in esc.get('fallas_por_recurso', {}).items():
                                            filas_falla_export.append({
                                                'Escenario': esc['nombre'],
                                                'Túneles Afectados': ", ".join(esc.get('frentes_aplicables', [])),
                                                'Recurso': recurso_f,
                                                'Probabilidad Falla (%)': v_f.get('probabilidad'),
                                                'Modo Demora': v_f.get('modo_demora', 'Manual'),
                                                'Demora Fija (h, si es manual)': v_f.get('demora'),
                                                'Distribución (si aplica)': v_f.get('distribucion'),
                                                'Parámetros Distribución': (
                                                    v_f.get('params_distribucion') if v_f.get('modo_demora') == 'Distribución estadística' else ''
                                                ),
                                            })
                                    if filas_falla_export:
                                        pd.DataFrame(filas_falla_export).to_excel(writer, sheet_name='Fallas Equipos', index=False)

                                    # Fallas REALMENTE ocurridas en la traza (una fila por falla, con
                                    # su demora ya sorteada), igual espiritu que 'Mantenciones Ejecutadas'.
                                    df_falla_traza = df_traza_completa[
                                        df_traza_completa['type'] == 'delay_equipment_failure'
                                    ].copy()
                                    if not df_falla_traza.empty:
                                        df_falla_ejecutada = df_falla_traza[[
                                            'Simulation_ID', 'Recurso', 'Resource_Origin_Tunnel', 'Cycle',
                                            'Start', 'Finish', 'Duration'
                                        ]].rename(columns={
                                            'Simulation_ID': 'Simulación', 'Recurso': 'Equipo',
                                            'Resource_Origin_Tunnel': 'Túnel', 'Cycle': 'Ciclo',
                                            'Start': 'Inicio (h)', 'Finish': 'Fin (h)', 'Duration': 'Duración (h)'
                                        })
                                        df_falla_ejecutada.to_excel(writer, sheet_name='Fallas Ejecutadas', index=False)
                                if st.session_state.get('mantenciones_programadas'):
                                    # Se exporta una version "aplanada" (sin el dict de params de
                                    # distribucion, que no cabe bien en una celda de Excel) mas legible
                                    # que un volcado directo del session_state.
                                    df_mant_export = pd.DataFrame([
                                        {
                                            'Nombre': m['nombre'],
                                            'Recurso': m['recurso'],
                                            'N° Unidad': m['instancia_id'] + 1,
                                            'Unidad de Tiempo': m['modo_tiempo'],
                                            'Inicio': m['inicio'],
                                            'Modo Duración': m['modo_duracion'],
                                            'Fin (si es manual)': m.get('fin'),
                                            'Distribución (si aplica)': m.get('distribucion'),
                                            'Parámetros Distribución': (
                                                m.get('params_distribucion') if m.get('modo_duracion') == 'Distribución estadística' else ''
                                            ),
                                            'Probabilidad Éxito (%)': m['probabilidad_exito'],
                                            'Modo Duración Reintento': m.get('modo_reintento', 'Igual a la mantención original'),
                                            'Distribución Reintento (si aplica)': m.get('distribucion_reintento'),
                                            'Parámetros Distribución Reintento': (
                                                m.get('params_distribucion_reintento')
                                                if m.get('modo_reintento') == 'Definir una distribución propia para el reintento' else ''
                                            ),
                                        }
                                        for m in st.session_state.mantenciones_programadas
                                    ])
                                    df_mant_export.to_excel(writer, sheet_name='Mantenciones Equipos', index=False)

                                    # HOJA: Resumen de mantenciones EJECUTADAS en la traza (una fila por
                                    # intento, exitoso o fallido), para poder auditar cuanto tiempo real
                                    # de indisponibilidad genero cada mantencion en cada simulacion.
                                    df_mant_traza = df_traza_completa[
                                        df_traza_completa['type'].isin(['maintenance', 'maintenance_failed'])
                                    ].copy()
                                    if not df_mant_traza.empty:
                                        df_mant_resumen = df_mant_traza[[
                                            'Simulation_ID', 'Recurso', 'Actividad', 'Start', 'Finish', 'Duration', 'type'
                                        ]].rename(columns={
                                            'Simulation_ID': 'Simulacion', 'Recurso': 'Equipo',
                                            'Actividad': 'Detalle', 'Start': 'Inicio (h)', 'Finish': 'Fin (h)',
                                            'Duration': 'Duración (h)', 'type': 'Resultado'
                                        })
                                        df_mant_resumen['Resultado'] = df_mant_resumen['Resultado'].map({
                                            'maintenance': 'Éxito', 'maintenance_failed': 'Falló (reintentó)'
                                        })
                                        df_mant_resumen.to_excel(writer, sheet_name='Mantenciones Ejecutadas', index=False)
                                if not st.session_state.get('df_marina', pd.DataFrame()).empty:
                                    st.session_state.df_marina.to_excel(writer, sheet_name='Vaciaderos Marina', index=False)
                                if not st.session_state.get('df_muckpile', pd.DataFrame()).empty:
                                    st.session_state.df_muckpile.to_excel(writer, sheet_name='Muckpile Stockpile', index=False)
                                if st.session_state.get('cambios_recursos'):
                                    pd.DataFrame(st.session_state.cambios_recursos).to_excel(writer, sheet_name='Cambios Recursos Turno', index=False)
                                if st.session_state.get('registro_cambios_flota'):
                                    pd.DataFrame(st.session_state.registro_cambios_flota).to_excel(writer, sheet_name='Cambios Flota Aplicados', index=False)

                                # HOJA 2: Avance Final por Tunel
                                df_resultados.to_excel(writer, sheet_name='Avance Final por Tunel', index=False)
                            
                                # HOJA 3: Avance por Turno
                                if datos_turno:
                                    df_turnos.to_excel(writer, sheet_name='Avance por Turno', index=False)
                            
                                # HOJA 4: Utilizacion de Recursos
                                if datos_recursos:
                                    df_recursos_res.to_excel(writer, sheet_name='Utilizacion de Recursos', index=False)
                            
                                # HOJA 5-N: Detalle por Simulacion y Tunel
                                for sim_id in sorted(df_traza_completa['Simulation_ID'].unique()):
                                    for tunel in tuneles_a_exportar:
                                        df_sim_tunel = df_traza_completa[
                                            (df_traza_completa['Simulation_ID'] == sim_id) &
                                            ((df_traza_completa['Resource_Origin_Tunnel'] == tunel) |
                                             (df_traza_completa['Resource_Destination_Tunnel'] == tunel))
                                        ].copy()
                                    
                                        if not df_sim_tunel.empty:
                                            df_sim_tunel = df_sim_tunel.sort_values(by='Start')
                                        
                                            # Crear hoja con nombre limitado
                                            sheet_name = f"S{sim_id}_{tunel}"[:31]
                                        
                                            df_export = df_sim_tunel[[
                                                'Cycle', 'Actividad', 'Recurso', 'type', 'Start', 'Finish', 'Duration',
                                                'Resource_Origin_Tunnel', 'Resource_Destination_Tunnel',
                                                'Travel_Distance_m', 'Travel_Speed_Used_m_h', 'Cycle_Advance_Length_m'
                                            ]].copy()
                                        
                                            df_export.columns = [
                                                'Ciclo', 'Actividad', 'Recurso', 'Tipo', 'Inicio (h)', 'Fin (h)', 
                                                'Duracion (h)', 'Tunel Origen', 'Tunel Destino', 'Distancia Viaje (m)',
                                                'Velocidad (m/h)', 'Avance Ciclo (m)'
                                            ]
                                        
                                            df_export.to_excel(writer, sheet_name=sheet_name, index=False)
                            
                                # HOJA: Resumen de Actividades por Tunel
                                resumen_actividades_data = []
                                for sim_id in sorted(df_traza_completa['Simulation_ID'].unique()):
                                    for tunel in tuneles_a_exportar:
                                        df_tunel_sim = df_traza_completa[
                                            (df_traza_completa['Simulation_ID'] == sim_id) &
                                            (df_traza_completa['Resource_Origin_Tunnel'] == tunel) &
                                            (df_traza_completa['type'].isin(['activity', 'delay_fh_oc', 'delay_res', 'delay_bloqueo_total', 'delay_equipment_failure', 'delay_fleet_change', 'delay_maintenance', 'delay_shift_change', 'blocked_no_resource']))
                                        ]
                                        # Mantenciones programadas de los equipos usados por el tunel, como actividad
                                        # (tramo que cae dentro de cada avance, igual que en el Menu 18).
                                        df_tunel_sim = incorporar_mantenciones_como_actividad(
                                            df_tunel_sim, df_traza_completa[df_traza_completa['Simulation_ID'] == sim_id], tunel
                                        )
                                        df_tunel_sim = df_tunel_sim[~df_tunel_sim['type'].isin(TIPOS_MANTENCION)]
                                    
                                        if not df_tunel_sim.empty:
                                            for actividad in df_tunel_sim['Actividad'].unique():
                                                df_act = df_tunel_sim[df_tunel_sim['Actividad'] == actividad]
                                                tiempo_total = df_act['Duration'].sum()
                                                num_ejecuciones = len(df_act)
                                                tiempo_promedio = tiempo_total / num_ejecuciones if num_ejecuciones > 0 else 0
                                            
                                                resumen_actividades_data.append({
                                                    'Simulacion': sim_id,
                                                    'Tunel': tunel,
                                                    'Actividad': actividad,
                                                    'Num Ejecuciones': num_ejecuciones,
                                                    'Tiempo Total (h)': round(tiempo_total, 2),
                                                    'Tiempo Promedio (h)': round(tiempo_promedio, 2)
                                                })
                            
                                if resumen_actividades_data:
                                    df_resumen_act = pd.DataFrame(resumen_actividades_data)
                                    df_resumen_act.to_excel(writer, sheet_name='Resumen Actividades', index=False)
                            
                                # HOJA: Resumen de Viajes por Recurso
                                resumen_viajes_data = []
                                for sim_id in sorted(df_traza_completa['Simulation_ID'].unique()):
                                    df_viajes_sim = df_traza_completa[
                                        (df_traza_completa['Simulation_ID'] == sim_id) &
                                        (df_traza_completa['type'] == 'travel')
                                    ].copy()
                                
                                    if not df_viajes_sim.empty:
                                        df_viajes_sim['Recurso_Tipo'] = df_viajes_sim['Recurso'].apply(
                                            lambda x: x.split('_')[0] if pd.notna(x) else 'N/A'
                                        )
                                    
                                        for recurso_tipo in df_viajes_sim['Recurso_Tipo'].unique():
                                            df_rec = df_viajes_sim[df_viajes_sim['Recurso_Tipo'] == recurso_tipo]
                                        
                                            resumen_viajes_data.append({
                                                'Simulacion': sim_id,
                                                'Recurso': recurso_tipo,
                                                'Num Viajes': len(df_rec),
                                                'Distancia Total (m)': round(df_rec['Travel_Distance_m'].sum(), 2),
                                                'Tiempo Total Viaje (h)': round(df_rec['Travel_Time_Actual'].sum(), 2),
                                                'Velocidad Promedio (m/h)': round(df_rec['Travel_Speed_Used_m_h'].mean(), 2)
                                            })
                            
                                if resumen_viajes_data:
                                    df_resumen_viajes = pd.DataFrame(resumen_viajes_data)
                                    df_resumen_viajes.to_excel(writer, sheet_name='Resumen Viajes', index=False)
                            
                                # HOJA: Ciclos Completados por Tunel
                                ciclos_data = []
                                for sim_id in sorted(df_traza_completa['Simulation_ID'].unique()):
                                    for tunel in tuneles_a_exportar:
                                        df_tunel_sim = df_traza_completa[
                                            (df_traza_completa['Simulation_ID'] == sim_id) &
                                            (df_traza_completa['Resource_Origin_Tunnel'] == tunel) &
                                            (df_traza_completa['type'] == 'activity')
                                        ]
                                    
                                        if not df_tunel_sim.empty:
                                            ciclos_completados = df_tunel_sim['Cycle'].max()
                                            avance_total = st.session_state.resultados.get(tunel, [0])[sim_id - 1] if sim_id <= len(st.session_state.resultados.get(tunel, [])) else 0
                                        
                                            ciclos_data.append({
                                                'Simulacion': sim_id,
                                                'Tunel': tunel,
                                                'Ciclos Completados': ciclos_completados,
                                                'Avance Total (m)': round(avance_total, 2),
                                                'Distancia Objetivo (m)': round(st.session_state.frentes_info[tunel]['distancia'], 2),
                                                'Porcentaje Completado (%)': round((avance_total / st.session_state.frentes_info[tunel]['distancia']) * 100, 2)
                                            })
                            
                                if ciclos_data:
                                    df_ciclos = pd.DataFrame(ciclos_data)
                                    df_ciclos.to_excel(writer, sheet_name='Ciclos Completados', index=False)
                            
                                # HOJA: Estadisticas de Demoras
                                demoras_data = []
                                for sim_id in sorted(df_traza_completa['Simulation_ID'].unique()):
                                    for tunel in tuneles_a_exportar:
                                        df_demoras = df_traza_completa[
                                            (df_traza_completa['Simulation_ID'] == sim_id) &
                                            (df_traza_completa['Resource_Origin_Tunnel'] == tunel) &
                                            (df_traza_completa['type'].isin(['delay_fh_oc', 'delay_res', 'delay_bloqueo_total', 'delay_equipment_failure', 'delay_fleet_change', 'delay_maintenance', 'delay_shift_change']))
                                        ]
                                    
                                        if not df_demoras.empty:
                                            for tipo_demora in df_demoras['type'].unique():
                                                df_tipo = df_demoras[df_demoras['type'] == tipo_demora]
                                            
                                                demoras_data.append({
                                                    'Simulacion': sim_id,
                                                    'Tunel': tunel,
                                                    'Tipo Demora': tipo_demora,
                                                    'Num Ocurrencias': len(df_tipo),
                                                    'Tiempo Total Demora (h)': round(df_tipo['Duration'].sum(), 2),
                                                    'Tiempo Promedio (h)': round(df_tipo['Duration'].mean(), 2)
                                                })
                            
                                if demoras_data:
                                    df_demoras_excel = pd.DataFrame(demoras_data)
                                    df_demoras_excel.to_excel(writer, sheet_name='Estadisticas Demoras', index=False)

                                # ---- HOJAS EQUIVALENTES A ARCHIVOS DAVID MONTENEGRO ----
                                # Advances_total_scenary (equivalente)
                                df_adv_total = generar_advances_total_scenary(
                                    st.session_state.resultados,
                                    st.session_state.frentes_info,
                                    st.session_state.get('nivel_simulado', nivel_seleccionado)
                                )
                                df_adv_total.to_excel(writer, sheet_name='Advances_Total_Scenary', index=False)

                                # Simulation_Front_Actual (una hoja por escenario, solo primeros 5 para el zip)
                                n_sim_exec = st.session_state.get('n_simulaciones_ejecutadas', 1)
                                dur_turno_exec = st.session_state.get('duracion_turno_simulada', st.session_state.get('duracion_turno_base', 8.0))
                                n_turnos_exec = st.session_state.get('numero_turnos_simulado', st.session_state.get('numero_turnos', 62))

                                sheets_front_act = generar_simulation_front_actual(
                                    st.session_state.traza_eventos,
                                    st.session_state.frentes_info,
                                    st.session_state.get('df_frentes_nivel_simuladas', frentes_nivel),
                                    min(n_sim_exec, 10)
                                )
                                for sname, sdf in sheets_front_act.items():
                                    sdf.to_excel(writer, sheet_name=f'FrontActual_{sname}'[:31], index=False)

                                # Time Program (primeros 5 escenarios para no sobrecargar)
                                sheets_time_prog = generar_simulation_time_program(
                                    st.session_state.traza_eventos,
                                    st.session_state.recursos_config,
                                    n_turnos_exec,
                                    dur_turno_exec,
                                    min(n_sim_exec, 5),
                                    st.session_state.get('ventanas_flota', {})
                                )
                                for sname, sdf in sheets_time_prog.items():
                                    sdf.to_excel(writer, sheet_name=f'TimeProgram_{sname}'[:31], index=False)

                            output.seek(0)

                            nombre_nivel = st.session_state.get('nivel_simulado', nivel_seleccionado)
                            st.download_button(
                                label="Descargar Reporte Excel Completo",
                                data=output,
                                file_name=f"Reporte_Simulacion_{nombre_nivel}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                type="primary",
                                width='stretch'
                            )

                            st.success("Reporte Excel generado. Haga clic en el botón para descargar.")
                            pass
                            contenido_reporte = [
                                "Resumen General — Parámetros de simulación",
                                "Avance Final por Tunel — P0, P10, P30, P50, Esperanza, P70, P90, P100 por frente",
                                "Avance por Turno — Progreso temporal acumulado",
                                "Utilizacion de Recursos — Uso y eficiencia",
                                "Detalle por Simulacion/Tunel — Eventos ordenados cronológicamente",
                                "Resumen Actividades — Tiempos por actividad y túnel",
                                "Resumen Viajes — Estadísticas de desplazamiento de recursos",
                                "Ciclos Completados — Avance y % de completitud",
                                "Estadisticas Demoras — Análisis de tiempos perdidos",
                                "Advances_Total_Scenary — Avance por frente/escenario (formato David Montenegro)",
                                "FrontActual_scenary N — Posiciones finales por escenario",
                                "TimeProgram_scenary N — Tiempo disponible por recurso/turno",
                                "Vaciaderos Marina — Sitios de depósito (si se cargó Input_Marina)",
                                "Muckpile Stockpile — Estado del stockpile (si se cargó Muckpile)",
                                "Cambios Recursos Turno — Cambios de cantidad de equipos por turno u hora"
                            ]
                            subtitulo_menu("Contenido del Reporte (hojas)",
                                           ayuda="\n".join(f"{i}. {item}" for i, item in enumerate(contenido_reporte, 1)))


# Dibuja la ventana flotante de logs con todo lo registrado en esta corrida.
_flujo_estado()['fin'] = True  # la corrida termino: los menus que no se dibujaron pasan a pendiente
_logs_refrescar(force=True)