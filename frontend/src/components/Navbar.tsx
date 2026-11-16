import React, { useState } from 'react';
import { Shield, Navigation, CheckCircle2, BarChart3, Upload, User as UserIcon, LogOut, Key, Camera, Activity, Cpu } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

interface NavbarProps {
  activeTab: 'map' | 'route' | 'verification' | 'analytics' | 'live_camera' | 'evaluation' | 'autonomous';
  setActiveTab: (tab: 'map' | 'route' | 'verification' | 'analytics' | 'live_camera' | 'evaluation' | 'autonomous') => void;
  onOpenUpload: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, setActiveTab, onOpenUpload }) => {
  const { user, logout, login } = useAuth();
  const [showLoginModal, setShowLoginModal] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loginError, setLoginError] = useState('');

  const handleLoginSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoginError('');
    try {
      await login(email, password);
      setShowLoginModal(false);
    } catch (err: any) {
      setLoginError(err.message || 'Login failed');
    }
  };

  const handleQuickDemoLogin = async (role: 'admin' | 'verifier' | 'contributor') => {
    try {
      if (role === 'admin') {
        await login('admin@accessguard.org', 'AccessGuard2026!');
      } else if (role === 'verifier') {
        await login('verifier@accessguard.org', 'Verifier2026!');
      } else {
        await login('contributor@accessguard.org', 'Contributor2026!');
      }
      setShowLoginModal(false);
    } catch (err: any) {
      setLoginError(err.message || 'Demo login failed');
    }
  };

  return (
    <>
      <header className="bg-slate-900 border-b border-slate-800 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <div className="flex items-center space-x-3 cursor-pointer" onClick={() => setActiveTab('map')}>
              <div className="bg-emerald-500/10 p-2 rounded-lg border border-emerald-500/30">
                <Shield className="w-6 h-6 text-emerald-400" />
              </div>
              <div>
                <span className="text-xl font-bold bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent">
                  AccessGuard
                </span>
                <span className="hidden sm:inline-block ml-2 text-xs font-semibold px-2 py-0.5 rounded-full bg-emerald-950 text-emerald-300 border border-emerald-800">
                  Public Benefit Intelligence
                </span>
              </div>
            </div>

            {/* Navigation Tabs */}
            <nav className="hidden md:flex space-x-1">
              <button
                onClick={() => setActiveTab('map')}
                className={`px-3 py-2 rounded-md text-sm font-medium transition-colors flex items-center space-x-2 ${
                  activeTab === 'map'
                    ? 'bg-slate-800 text-emerald-400 border border-slate-700'
                    : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
                }`}
              >
                <Shield className="w-4 h-4" />
                <span>Map Dashboard</span>
              </button>

              <button
                onClick={() => setActiveTab('route')}
                className={`px-3 py-2 rounded-md text-sm font-medium transition-colors flex items-center space-x-2 ${
                  activeTab === 'route'
                    ? 'bg-slate-800 text-emerald-400 border border-slate-700'
                    : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
                }`}
              >
                <Navigation className="w-4 h-4" />
                <span>Accessible Routing</span>
              </button>

              <button
                onClick={() => setActiveTab('verification')}
                className={`px-3 py-2 rounded-md text-sm font-medium transition-colors flex items-center space-x-2 ${
                  activeTab === 'verification'
                    ? 'bg-slate-800 text-emerald-400 border border-slate-700'
                    : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
                }`}
              >
                <CheckCircle2 className="w-4 h-4" />
                <span>Verification Queue</span>
              </button>

              <button
                onClick={() => setActiveTab('analytics')}
                className={`px-3 py-2 rounded-md text-sm font-medium transition-colors flex items-center space-x-2 ${
                  activeTab === 'analytics'
                    ? 'bg-slate-800 text-emerald-400 border border-slate-700'
                    : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
                }`}
              >
                <BarChart3 className="w-4 h-4" />
                <span>Analytics</span>
              </button>

              <button
                onClick={() => setActiveTab('live_camera')}
                className={`px-3 py-2 rounded-md text-sm font-medium transition-colors flex items-center space-x-2 ${
                  activeTab === 'live_camera'
                    ? 'bg-slate-800 text-emerald-400 border border-slate-700'
                    : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
                }`}
              >
                <Camera className="w-4 h-4" />
                <span>Live Camera</span>
              </button>

              <button
                onClick={() => setActiveTab('evaluation')}
                className={`px-3 py-2 rounded-md text-sm font-medium transition-colors flex items-center space-x-2 ${
                  activeTab === 'evaluation'
                    ? 'bg-slate-800 text-emerald-400 border border-slate-700'
                    : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
                }`}
              >
                <Activity className="w-4 h-4" />
                <span>Evaluation & Datasets</span>
              </button>

              <button
                onClick={() => setActiveTab('autonomous')}
                className={`px-3 py-2 rounded-md text-sm font-medium transition-colors flex items-center space-x-2 ${
                  activeTab === 'autonomous'
                    ? 'bg-slate-800 text-indigo-400 border border-indigo-700/50'
                    : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
                }`}
              >
                <Cpu className="w-4 h-4" />
                <span>Autonomous AI</span>
              </button>
            </nav>

            {/* Actions: Upload CTA & Auth */}
            <div className="flex items-center space-x-3">
              <button
                onClick={onOpenUpload}
                className="bg-emerald-600 hover:bg-emerald-500 text-white px-3.5 py-2 rounded-lg text-sm font-semibold flex items-center space-x-1.5 shadow-lg shadow-emerald-900/30 transition-all hover:scale-[1.02]"
              >
                <Upload className="w-4 h-4" />
                <span>Upload Observation</span>
              </button>

              {user ? (
                <div className="flex items-center space-x-2 bg-slate-800/80 px-3 py-1.5 rounded-lg border border-slate-700">
                  <div className="text-right hidden sm:block">
                    <div className="text-xs font-semibold text-slate-200">{user.email.split('@')[0]}</div>
                    <div className="text-[10px] uppercase font-bold text-emerald-400 tracking-wider">{user.role}</div>
                  </div>
                  <button
                    onClick={logout}
                    title="Log Out"
                    className="p-1 text-slate-400 hover:text-rose-400 transition-colors"
                  >
                    <LogOut className="w-4 h-4" />
                  </button>
                </div>
              ) : (
                <button
                  onClick={() => setShowLoginModal(true)}
                  className="bg-slate-800 hover:bg-slate-700 text-slate-200 px-3 py-2 rounded-lg text-sm font-medium border border-slate-700 flex items-center space-x-1.5 transition-colors"
                >
                  <UserIcon className="w-4 h-4" />
                  <span>Sign In</span>
                </button>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Login Modal */}
      {showLoginModal && (
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-700 rounded-xl max-w-md w-full p-6 shadow-2xl relative">
            <h2 className="text-xl font-bold text-white mb-2 flex items-center space-x-2">
              <Key className="w-5 h-5 text-emerald-400" />
              <span>Sign In to AccessGuard</span>
            </h2>
            <p className="text-sm text-slate-400 mb-4">
              Authenticate to submit human verifications and review sidewalk accessibility data.
            </p>

            {loginError && (
              <div className="mb-4 p-3 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-400 text-sm">
                {loginError}
              </div>
            )}

            <form onSubmit={handleLoginSubmit} className="space-y-4">
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">Email Address</label>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  placeholder="name@example.com"
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-emerald-500"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">Password</label>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  placeholder="••••••••"
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-emerald-500"
                />
              </div>

              <div className="flex justify-end space-x-2 pt-2">
                <button
                  type="button"
                  onClick={() => setShowLoginModal(false)}
                  className="px-4 py-2 text-sm text-slate-400 hover:text-white"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg text-sm font-semibold transition-colors"
                >
                  Sign In
                </button>
              </div>
            </form>

            <div className="mt-6 pt-4 border-t border-slate-800 text-center">
              <p className="text-xs text-slate-500 mb-2">Quick Demo Accounts:</p>
              <div className="flex justify-center space-x-2">
                <button
                  onClick={() => handleQuickDemoLogin('admin')}
                  className="text-xs bg-slate-800 hover:bg-slate-700 px-2.5 py-1 rounded border border-slate-700 text-emerald-400"
                >
                  Admin
                </button>
                <button
                  onClick={() => handleQuickDemoLogin('verifier')}
                  className="text-xs bg-slate-800 hover:bg-slate-700 px-2.5 py-1 rounded border border-slate-700 text-cyan-400"
                >
                  Verifier
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
};
