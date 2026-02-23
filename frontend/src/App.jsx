import { useState, useRef, useCallback } from 'react';
import axios from 'axios';

const API_BASE = '/api';

export default function App() {
  const [sampleFile, setSampleFile] = useState(null);
  const [sampleInfo, setSampleInfo] = useState(null);
  const [text, setText] = useState('');
  const [language, setLanguage] = useState('en');
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef(null);

  const handleUpload = useCallback(async (file) => {
    if (!file) return;
    setUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const { data } = await axios.post(`${API_BASE}/upload`, formData);
      setSampleFile(file);
      setSampleInfo(data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Upload failed');
    } finally {
      setUploading(false);
    }
  }, []);

  const handleGenerate = useCallback(async () => {
    if (!sampleFile || !text.trim()) return;
    setLoading(true);
    setError(null);
    setAudioUrl(null);

    const formData = new FormData();
    formData.append('file', sampleFile);
    formData.append('text', text);
    formData.append('language', language);

    try {
      const { data } = await axios.post(`${API_BASE}/generate`, formData, {
        responseType: 'blob',
      });
      const url = URL.createObjectURL(data);
      setAudioUrl(url);
    } catch (err) {
      if (err.response?.data instanceof Blob) {
        const text = await err.response.data.text();
        try {
          const json = JSON.parse(text);
          setError(json.detail || 'Generation failed');
        } catch {
          setError('Generation failed');
        }
      } else {
        setError(err.response?.data?.detail || err.message || 'Generation failed');
      }
    } finally {
      setLoading(false);
    }
  }, [sampleFile, text, language]);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleUpload(file);
  }, [handleUpload]);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setDragOver(false);
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-gray-950">
      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-950/80 backdrop-blur-sm">
        <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-brand-500 to-brand-700 rounded-xl flex items-center justify-center">
              <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
              </svg>
            </div>
            <h1 className="text-xl font-bold bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
              VocalClone
            </h1>
          </div>
          <span className="text-xs text-gray-500 bg-gray-800 px-3 py-1 rounded-full">
            Creator Dashboard
          </span>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-6 py-10 space-y-8">
        {/* Error Banner */}
        {error && (
          <div className="bg-red-900/30 border border-red-800 rounded-xl p-4 flex items-center gap-3">
            <svg className="w-5 h-5 text-red-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p className="text-red-300 text-sm">{error}</p>
            <button onClick={() => setError(null)} className="ml-auto text-red-400 hover:text-red-300">
              ✕
            </button>
          </div>
        )}

        {/* Step 1: Upload Voice Sample */}
        <section className="bg-gray-900/50 border border-gray-800 rounded-2xl p-6">
          <div className="flex items-center gap-2 mb-4">
            <span className="w-7 h-7 bg-brand-600 rounded-lg flex items-center justify-center text-sm font-bold">1</span>
            <h2 className="text-lg font-semibold">Upload Voice Sample</h2>
            <span className="text-xs text-gray-500 ml-2">WAV or MP3 · 1-30 seconds</span>
          </div>

          {!sampleInfo ? (
            <div
              className={`drop-zone ${dragOver ? 'drag-over' : ''}`}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".wav,.mp3"
                className="hidden"
                onChange={(e) => handleUpload(e.target.files[0])}
              />
              {uploading ? (
                <div className="flex flex-col items-center gap-3">
                  <div className="w-10 h-10 border-2 border-brand-500 border-t-transparent rounded-full animate-spin" />
                  <p className="text-gray-400">Processing audio...</p>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-3">
                  <svg className="w-12 h-12 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                      d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                  <p className="text-gray-400">
                    <span className="text-brand-400 font-medium">Click to upload</span> or drag and drop
                  </p>
                  <p className="text-xs text-gray-600">Record a 10-second sample of your voice</p>
                </div>
              )}
            </div>
          ) : (
            <div className="flex items-center gap-4 bg-gray-800/50 rounded-xl p-4">
              <div className="w-12 h-12 bg-green-900/30 rounded-xl flex items-center justify-center">
                <svg className="w-6 h-6 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <div className="flex-1">
                <p className="font-medium">{sampleInfo.filename}</p>
                <p className="text-sm text-gray-400">{sampleInfo.duration}s duration</p>
              </div>
              <button
                onClick={() => { setSampleFile(null); setSampleInfo(null); setAudioUrl(null); }}
                className="text-gray-400 hover:text-white text-sm px-3 py-1 border border-gray-700 rounded-lg hover:border-gray-500 transition"
              >
                Replace
              </button>
            </div>
          )}
        </section>

        {/* Step 2: Enter Script */}
        <section className="bg-gray-900/50 border border-gray-800 rounded-2xl p-6">
          <div className="flex items-center gap-2 mb-4">
            <span className="w-7 h-7 bg-brand-600 rounded-lg flex items-center justify-center text-sm font-bold">2</span>
            <h2 className="text-lg font-semibold">Enter Your Script</h2>
          </div>

          {/* Language Toggle */}
          <div className="flex gap-2 mb-4">
            <button
              onClick={() => setLanguage('en')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition ${
                language === 'en'
                  ? 'bg-brand-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:text-white'
              }`}
            >
              🇺🇸 English
            </button>
            <button
              onClick={() => setLanguage('ur')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition ${
                language === 'ur'
                  ? 'bg-brand-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:text-white'
              }`}
            >
              🇵🇰 اردو (Urdu)
            </button>
          </div>

          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder={language === 'en'
              ? 'Type your script here... e.g., "Welcome to my channel!"'
              : 'اپنا متن یہاں لکھیں...'
            }
            dir={language === 'ur' ? 'rtl' : 'ltr'}
            className="w-full h-36 bg-gray-800 border border-gray-700 rounded-xl p-4 text-white
                       placeholder-gray-500 resize-none focus:outline-none focus:border-brand-500
                       focus:ring-1 focus:ring-brand-500 transition"
            maxLength={5000}
          />
          <div className="flex justify-between mt-2">
            <p className="text-xs text-gray-600">
              {language === 'ur' ? 'اردو کی منفرد آوازوں کو بہتر بنایا گیا ہے' : 'Optimized for natural-sounding output'}
            </p>
            <p className="text-xs text-gray-500">{text.length}/5000</p>
          </div>
        </section>

        {/* Step 3: Generate */}
        <section className="bg-gray-900/50 border border-gray-800 rounded-2xl p-6">
          <div className="flex items-center gap-2 mb-4">
            <span className="w-7 h-7 bg-brand-600 rounded-lg flex items-center justify-center text-sm font-bold">3</span>
            <h2 className="text-lg font-semibold">Generate Cloned Audio</h2>
          </div>

          <button
            onClick={handleGenerate}
            disabled={!sampleFile || !text.trim() || loading}
            className="w-full py-3 bg-gradient-to-r from-brand-600 to-brand-700 hover:from-brand-500
                       hover:to-brand-600 disabled:from-gray-700 disabled:to-gray-700 disabled:cursor-not-allowed
                       rounded-xl font-semibold transition-all duration-300 flex items-center justify-center gap-2"
          >
            {loading ? (
              <>
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Generating...
              </>
            ) : (
              <>
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                Generate Speech
              </>
            )}
          </button>

          {/* Audio Player */}
          {audioUrl && (
            <div className="mt-6 bg-gray-800/50 rounded-xl p-5">
              <div className="flex items-center gap-3 mb-3">
                <div className="w-8 h-8 bg-green-900/30 rounded-lg flex items-center justify-center">
                  <svg className="w-4 h-4 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                      d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
                  </svg>
                </div>
                <p className="font-medium text-sm">Generated Audio</p>
                <span className="text-xs bg-brand-600/20 text-brand-300 px-2 py-0.5 rounded-full">
                  {language === 'en' ? 'English' : 'Urdu'}
                </span>
              </div>
              <audio controls src={audioUrl} className="w-full" />
              <div className="mt-3 flex gap-2">
                <a
                  href={audioUrl}
                  download={`vocalclone_${language}_output.wav`}
                  className="text-sm text-brand-400 hover:text-brand-300 flex items-center gap-1"
                >
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                      d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  Download
                </a>
              </div>
            </div>
          )}
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-800 mt-16">
        <div className="max-w-5xl mx-auto px-6 py-6 text-center text-xs text-gray-600">
          VocalClone · AI Voice Cloning for Content Creators · English &amp; Urdu
        </div>
      </footer>
    </div>
  );
}
