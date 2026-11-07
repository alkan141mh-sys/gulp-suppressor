// Popup: settings live in chrome.storage.sync (the content script listens for changes);
// status comes from the content script of the active tab.
const DEFAULTS = { enabled: true, fa: '2', atten: 30 };
const $ = (id) => document.getElementById(id);

async function load() {
  const s = { ...DEFAULTS, ...(await chrome.storage.sync.get(Object.keys(DEFAULTS))) };
  $('enabled').checked = s.enabled;
  // A setting saved by an older version (5/10/15/20 measured alarms) no longer exists: use the default.
  const known = [...$('fa').options].map(o => o.value);
  if (!known.includes(String(s.fa))) { s.fa = DEFAULTS.fa; chrome.storage.sync.set({ fa: s.fa }); }
  $('fa').value = String(s.fa);
  $('atten').value = String(s.atten);
}

$('enabled').onchange = () => chrome.storage.sync.set({ enabled: $('enabled').checked });
$('fa').onchange = () => chrome.storage.sync.set({ fa: $('fa').value });
$('atten').onchange = () => chrome.storage.sync.set({ atten: Number($('atten').value) });

async function status() {
  // tab.url is empty without the "tabs" permission, so the content script itself is asked: only
  // YouTube pages have one, and no answer means this is not a (freshly loaded) YouTube page.
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) return;
  try {
    const r = await chrome.tabs.sendMessage(tab.id, { type: 'getStats' });
    if (r && !r.allowed) {
      $('status').textContent = r.watch ? 'Bu video Buders kanalından değil: eklenti bu videoda kapalı.'
        : 'Eklenti yalnızca Buders videolarında çalışır. Bir Buders videosu aç.';
      return;
    }
    if (!r || !r.attached) {
      $('status').textContent = 'Video bulunamadı. Eklentiyi yeni kurduysan sayfayı yenile.';
      return;
    }
    const st = r.state === 'running' ? 'çalışıyor' : 'beklemede (videoyu oynat)';
    const a = r.ahead;
    const how = a && a.ahead
      ? `önden okuma (${a.aheadS === null ? 'bu an henüz çözülmedi' : `${Math.max(0, a.aheadS).toFixed(0)} s ileri analiz edildi`}` +
        `, saat farkı ${a.offsetMs} ms${a.calR !== null ? `, r = ${a.calR}` : ''})`
      : 'canlı mod (115 ms gecikmeli)';
    $('status').innerHTML = `<b>${st}</b> · ${how}<br>Bu sayfada kısılan ses: <b>${r.nMutes ?? 0}</b>` +
      (r.late ? ` · geç kalan karar: ${r.late}` : '') +
      (a && a.decodeErrors ? ` · çözme hatası: ${a.decodeErrors}` : '') +
      (a ? `<br><span style="color:#888;font-size:11px">tanı: t=${a.t} s · çözülen bölümler ${a.near.join(', ') || 'yok'} · ` +
           `ekleme ${a.appends}, paket ${a.frames}, çözülen ${(a.decoded / 48000).toFixed(0)} s, tekrar ${a.duplicates}, ` +
           `demux hata ${a.demuxErrors} (${a.demuxState}), decoder ${a.hasDecoder ? 'var' : 'yok'}, ${a.sbType || 'akış yok'}</span>` : '');
  } catch {
    $('status').textContent = 'Bu sekmede eklenti çalışmıyor. YouTube sekmesiysen sayfayı yenile (F5).';
  }
}

load();
status();
setInterval(status, 1000);
