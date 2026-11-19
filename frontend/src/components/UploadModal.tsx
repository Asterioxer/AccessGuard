import React, { useState, useRef } from 'react';
import { X, Upload, Image as ImageIcon, Video, MapPin, Cpu, AlertCircle, CheckCircle } from 'lucide-react';
import { api } from '../api/client';
import { Observation } from '../types';

interface UploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onUploadSuccess: (observation: Observation) => void;
}

export const UploadModal: React.FC<UploadModalProps> = ({ isOpen, onClose, onUploadSuccess }) => {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [latitude, setLatitude] = useState<string>('');
  const [longitude, setLongitude] = useState<string>('');
  const [adapter, setAdapter] = useState<string>('unified');
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  if (!isOpen) return null;

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selected = e.target.files[0];
      setFile(selected);
      setError(null);
      if (selected.type.startsWith('image/')) {
        setPreviewUrl(URL.createObjectURL(selected));
      } else {
        setPreviewUrl(null);
      }
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const selected = e.dataTransfer.files[0];
      setFile(selected);
      setError(null);
      if (selected.type.startsWith('image/')) {
        setPreviewUrl(URL.createObjectURL(selected));
      } else {
        setPreviewUrl(null);
      }
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      setError('Please select an image or video file.');
      return;
    }

    setIsUploading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);
      if (latitude) formData.append('latitude', latitude);
      if (longitude) formData.append('longitude', longitude);
      if (adapter) formData.append('adapter', adapter);

      const result = await api.uploadObservation(formData);
      setSuccess(true);
      setTimeout(() => {
        onUploadSuccess(result);
        onClose();
      }, 1200);
    } catch (err: any) {
      setError(err.message || 'Upload and inference failed.');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-700 rounded-2xl max-w-xl w-full p-6 shadow-2xl relative max-h-[90vh] overflow-y-auto">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-slate-400 hover:text-white transition-colors"
        >
          <X className="w-5 h-5" />
        </button>

        <div className="flex items-center space-x-2.5 mb-2">
          <div className="p-2 bg-emerald-500/10 rounded-lg border border-emerald-500/30">
            <Upload className="w-5 h-5 text-emerald-400" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-white">Upload Sidewalk Observation</h2>
            <p className="text-xs text-slate-400">Street-level image or short video analysis</p>
          </div>
        </div>

        {error && (
          <div className="mb-4 p-3 bg-rose-500/10 border border-rose-500/30 text-rose-400 rounded-xl text-xs flex items-center space-x-2">
            <AlertCircle className="w-4 h-4 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {success && (
          <div className="mb-4 p-3 bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 rounded-xl text-xs flex items-center space-x-2">
            <CheckCircle className="w-4 h-4 flex-shrink-0" />
            <span>Inference complete! Barrier analyzed and mapped to sidewalk segment.</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Drag & Drop Zone */}
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-colors ${
              file ? 'border-emerald-500/50 bg-emerald-950/10' : 'border-slate-700 hover:border-slate-600 bg-slate-800/40'
            }`}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept="image/jpeg,image/png,image/webp,video/mp4,video/quicktime"
              className="hidden"
            />

            {previewUrl ? (
              <div className="space-y-2">
                <img
                  src={previewUrl}
                  alt="Preview"
                  className="max-h-48 mx-auto rounded-lg object-cover border border-slate-700"
                />
                <p className="text-xs text-slate-400">{file?.name} (Click to change)</p>
              </div>
            ) : file ? (
              <div className="space-y-2">
                <Video className="w-10 h-10 text-emerald-400 mx-auto" />
                <p className="text-sm font-medium text-slate-200">{file.name}</p>
                <p className="text-xs text-slate-400">Video selected (Click to change)</p>
              </div>
            ) : (
              <div className="space-y-2">
                <div className="flex justify-center space-x-2 text-slate-400">
                  <ImageIcon className="w-8 h-8" />
                  <Video className="w-8 h-8" />
                </div>
                <p className="text-sm font-medium text-slate-300">
                  Drop street photo or short video here, or <span className="text-emerald-400 underline">browse</span>
                </p>
                <p className="text-[11px] text-slate-500">
                  Supports JPEG, PNG, MP4 (Max 20MB photos, 100MB videos). EXIF coordinates extracted automatically.
                </p>
              </div>
            )}
          </div>

          {/* Model Adapter Selection */}
          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1 flex items-center space-x-1.5">
              <Cpu className="w-3.5 h-3.5 text-cyan-400" />
              <span>Interchangeable CV Adapter</span>
            </label>
            <select
              value={adapter}
              onChange={(e) => setAdapter(e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white text-xs focus:outline-none focus:border-emerald-500"
            >
              <option value="unified">Unified Inference Engine (Hybrid Vision V2 + Multi-Specialist)</option>
              <option value="heuristic">OpenCV Heuristic Pipeline (Development Fallback Mode)</option>
              <option value="yolo">YOLO / ONNX Specialist Adapter</option>
              <option value="torchvision">PyTorch Torchvision Mask R-CNN Adapter</option>
              <option value="dummy">Deterministic Benchmark Adapter</option>
            </select>
          </div>

          {/* Geographic Coordinates */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1 flex items-center space-x-1">
                <MapPin className="w-3 h-3 text-emerald-400" />
                <span>Latitude</span>
              </label>
              <input
                type="number"
                step="any"
                value={latitude}
                onChange={(e) => setLatitude(e.target.value)}
                placeholder="37.7765"
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white text-xs focus:outline-none focus:border-emerald-500"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1 flex items-center space-x-1">
                <MapPin className="w-3 h-3 text-emerald-400" />
                <span>Longitude</span>
              </label>
              <input
                type="number"
                step="any"
                value={longitude}
                onChange={(e) => setLongitude(e.target.value)}
                placeholder="-122.4194"
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white text-xs focus:outline-none focus:border-emerald-500"
              />
            </div>
          </div>

          <p className="text-[11px] text-slate-500">
            Privacy Protection: Faces and vehicle plates are blurred automatically. EXIF metadata is permanently stripped before storage.
          </p>

          <div className="flex justify-end space-x-2 pt-2">
            <button
              type="button"
              onClick={onClose}
              disabled={isUploading}
              className="px-4 py-2 text-xs text-slate-400 hover:text-white"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isUploading || !file}
              className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white rounded-lg text-xs font-semibold transition-colors flex items-center space-x-1.5 shadow-lg shadow-emerald-900/30"
            >
              {isUploading ? (
                <span>Analyzing Geometry & Barriers...</span>
              ) : (
                <>
                  <Upload className="w-3.5 h-3.5" />
                  <span>Run Inference & Submit</span>
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
