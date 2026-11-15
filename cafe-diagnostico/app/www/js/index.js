(function () {
    'use strict';

    var $ = function (id) { return document.getElementById(id); };
    var REASONS = {
        baixa_confianca: 'O modelo não tem confiança suficiente neste caso.',
        imagem_borrada: 'A foto está borrada ou com pouca nitidez. Tente novamente com a folha em foco.'
    };

    var currentBlob = null;
    var currentResult = null;

    document.addEventListener('deviceready', init, false);

    function init() {
        try { $('api').value = localStorage.getItem('api') || ''; } catch (e) { /* sem storage */ }
        if (!$('api').value) { $('api').value = 'http://192.168.0.10:8000'; }

        $('btn-camera').onclick = function () { pick(Camera.PictureSourceType.CAMERA); };
        $('btn-gallery').onclick = function () { pick(Camera.PictureSourceType.PHOTOLIBRARY); };
        $('btn-test').onclick = testConnection;
        $('btn-escalate').onclick = escalate;
    }

    function apiUrl() {
        var v = $('api').value.trim().replace(/\/+$/, '');
        try { localStorage.setItem('api', v); } catch (e) { /* ignora */ }
        return v;
    }

    function testConnection() {
        var c = $('conn');
        c.className = 'small';
        c.textContent = 'Testando…';
        fetchWithTimeout(apiUrl() + '/health', {}, 8000)
            .then(function (r) { return r.json(); })
            .then(function (j) {
                c.className = 'small ok';
                c.textContent = 'Conectado ao servidor.';
            })
            .catch(function () {
                c.className = 'small fail';
                c.textContent = 'Não foi possível conectar. Confira o endereço, o Wi-Fi e o firewall.';
            });
    }

    function pick(source) {
        navigator.camera.getPicture(onPhoto, function (msg) {
            if (msg && !/cancel|no image|selected/i.test(String(msg))) { showError(String(msg)); }
        }, {
            quality: 85,
            destinationType: Camera.DestinationType.DATA_URL,
            sourceType: source,
            encodingType: Camera.EncodingType.JPEG,
            mediaType: Camera.MediaType.PICTURE,
            targetWidth: 2048,
            targetHeight: 2048,
            correctOrientation: true,
            saveToPhotoAlbum: false
        });
    }

    // Aceita base64 puro, "data:...;base64,..." ou um endereço (file:, content:, blob:, http:)
    function toBlob(data) {
        data = String(data || '');
        if (/^(file|content|blob|https?):/i.test(data)) {
            return fetch(data).then(function (r) { return r.blob(); });
        }
        var b64 = data.replace(/^data:[^,]*,/, '').replace(/\s+/g, '')
            .replace(/-/g, '+').replace(/_/g, '/');
        while (b64.length % 4) { b64 += '='; }
        var bin = atob(b64);
        var arr = new Uint8Array(bin.length);
        for (var i = 0; i < bin.length; i++) { arr[i] = bin.charCodeAt(i); }
        return Promise.resolve(new Blob([arr], { type: 'image/jpeg' }));
    }

    function fetchWithTimeout(url, opts, ms) {
        var ctrl = new AbortController();
        var t = setTimeout(function () { ctrl.abort(); }, ms);
        opts = opts || {};
        opts.signal = ctrl.signal;
        return fetch(url, opts).then(function (r) { clearTimeout(t); return r; },
            function (e) { clearTimeout(t); throw e; });
    }

    function onPhoto(data) {
        currentBlob = null;
        currentResult = null;
        hide('error'); hide('result');
        show('loading');
        setButtons(true);

        toBlob(data)
            .then(function (blob) {
                currentBlob = blob;
                var fd = new FormData();
                fd.append('file', blob, 'folha.jpg');
                return fetchWithTimeout(apiUrl() + '/predict',
                    { method: 'POST', body: fd }, 90000);
            })
            .then(function (r) {
                if (!r.ok) {
                    return r.text().then(function (t) { throw new Error('Servidor respondeu ' + r.status + ': ' + t); });
                }
                return r.json();
            })
            .then(function (res) { currentResult = res; render(res); })
            .catch(function (e) {
                var msg = (e && e.name === 'AbortError')
                    ? 'A análise demorou demais. Tente novamente.'
                    : 'Falha na comunicação com o servidor. Verifique o endereço e o Wi-Fi. (' + (e && e.message ? e.message : e) + ')';
                showError(msg);
            })
            .then(function () { hide('loading'); setButtons(false); });
    }

    function render(r) {
        var refused = r.refused;
        var pct = function (x) { return (x * 100).toFixed(1) + '%'; };

        $('banner').className = 'banner' + (refused ? ' refused' : '');
        if (refused) {
            $('title').textContent = 'Caso incerto – enviar para o agrônomo';
            $('subtitle').textContent = (REASONS[r.reason] || '') +
                ' Palpite do modelo: ' + r.class_pt + ' (' + pct(r.confidence) + ').';
        } else {
            $('title').textContent = r.class_pt;
            $('subtitle').textContent = 'Confiança do modelo: ' + pct(r.confidence);
        }

        $('cam').src = 'data:image/jpeg;base64,' + r.gradcam;

        var box = $('probs');
        box.textContent = '';
        Object.keys(r.probs)
            .sort(function (a, b) { return r.probs[b] - r.probs[a]; })
            .forEach(function (name, i) {
                var row = document.createElement('div');
                row.className = 'bar' + (i === 0 ? ' top1' : '');
                var top = document.createElement('div');
                top.className = 'top';
                var n = document.createElement('span');
                n.textContent = name;
                var v = document.createElement('span');
                v.textContent = pct(r.probs[name]);
                top.appendChild(n); top.appendChild(v);
                var track = document.createElement('div');
                track.className = 'track';
                var fill = document.createElement('div');
                fill.className = 'fill';
                fill.style.width = (r.probs[name] * 100).toFixed(1) + '%';
                track.appendChild(fill);
                row.appendChild(top); row.appendChild(track);
                box.appendChild(row);
            });

        var btn = $('btn-escalate');
        btn.className = refused ? 'highlight' : '';
        btn.disabled = false;
        hide('sent');
        show('result');
        window.scrollTo(0, 0);
    }

    function escalate() {
        if (!currentBlob || !currentResult) { return; }
        var btn = $('btn-escalate');
        btn.disabled = true;

        var fd = new FormData();
        fd.append('file', currentBlob, 'folha.jpg');
        fd.append('predicted', currentResult['class']);
        fd.append('confidence', String(currentResult.confidence));
        fd.append('reason', currentResult.reason || '');

        fetchWithTimeout(apiUrl() + '/escalate', { method: 'POST', body: fd }, 30000)
            .then(function (r) { if (!r.ok) { throw new Error('status ' + r.status); } return r.json(); })
            .then(function () {
                $('sent').textContent = 'Foto e diagnóstico enviados ao agrônomo.';
                show('sent');
            })
            .catch(function () {
                btn.disabled = false;
                $('sent').textContent = 'Não foi possível enviar. Tente novamente.';
                show('sent');
            });
    }

    function setButtons(disabled) {
        $('btn-camera').disabled = disabled;
        $('btn-gallery').disabled = disabled;
    }
    function showError(msg) { $('error').textContent = msg; show('error'); }
    function show(id) { $(id).hidden = false; }
    function hide(id) { $(id).hidden = true; }
})();
