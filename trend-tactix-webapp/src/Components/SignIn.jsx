// SignIn.jsx - Redesigned to match sidebar styling and fully responsive
import React, { useState } from 'react';
import { Eye, EyeOff, LogIn, BarChart2, Package, Users, AlertCircle } from 'lucide-react';

const Signin = ({ onSignIn }) => {
  const [showPassword, setShowPassword] = useState(false);
  const [credentials, setCredentials] = useState({
    username: '',
    password: ''
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    // Simple hardcoded authentication
    if (credentials.username === 'saif' && credentials.password === '12345') {
      setTimeout(() => {
        onSignIn();
        setLoading(false);
      }, 1000); // Simulate loading
    } else {
      setError('Invalid username or password');
      setLoading(false);
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setCredentials(prev => ({
      ...prev,
      [name]: value
    }));
    setError(''); // Clear error when user starts typing
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-indigo-800 to-indigo-900 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Logo Section - Matching Sidebar */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center mb-6">
            <div className="w-16 h-16 sm:w-20 sm:h-20 bg-gradient-to-b from-indigo-800 to-indigo-900 rounded-xl flex items-center justify-center border border-indigo-700 shadow-lg">
              <span className="text-2xl sm:text-3xl font-bold text-white">TT</span>
            </div>
          </div>
          <h1 className="text-3xl sm:text-4xl font-bold text-white mb-2">Trend Tactix</h1>
          <p className="text-base sm:text-lg text-indigo-200">Fashion Inventory Management</p>
        </div>

        {/* Features Icons - Responsive */}
        <div className="bg-white/5 backdrop-blur-lg rounded-lg p-4 mb-8 border border-white/10">
          <div className="grid grid-cols-3 gap-3 sm:gap-4 text-center">
            <div className="flex flex-col items-center">
              <BarChart2 className="w-6 h-6 sm:w-8 sm:h-8 text-indigo-300 mb-2" />
              <span className="text-xs sm:text-sm text-indigo-200">Analytics</span>
            </div>
            <div className="flex flex-col items-center">
              <Package className="w-6 h-6 sm:w-8 sm:h-8 text-indigo-300 mb-2" />
              <span className="text-xs sm:text-sm text-indigo-200">Inventory</span>
            </div>
            <div className="flex flex-col items-center">
              <Users className="w-6 h-6 sm:w-8 sm:h-8 text-indigo-300 mb-2" />
              <span className="text-xs sm:text-sm text-indigo-200">Multi-User</span>
            </div>
          </div>
        </div>

        {/* Login Form - Fully Responsive */}
        <div className="bg-white/10 backdrop-blur-lg rounded-xl shadow-xl p-6 sm:p-8 border border-white/10">
          <div className="text-center mb-6">
            <h2 className="text-xl sm:text-2xl font-semibold text-white">Welcome Back</h2>
            <p className="text-sm sm:text-base text-indigo-200 mt-2">Sign in to your account</p>
          </div>

          <form className="space-y-6" onSubmit={handleSubmit}>
            {/* Username Field */}
            <div>
              <label htmlFor="username" className="block text-sm font-medium text-indigo-100 mb-2">
                Username
              </label>
              <input
                id="username"
                name="username"
                type="text"
                required
                value={credentials.username}
                onChange={handleInputChange}
                className="w-full px-4 py-3 border border-white/20 rounded-lg bg-white/5 text-white placeholder-indigo-300 focus:outline-none focus:ring-2 focus:ring-white/30 focus:border-white/40 transition duration-200 text-sm sm:text-base"
                placeholder="Enter your username"
                autoFocus
              />
            </div>

            {/* Password Field */}
            <div>
              <label htmlFor="password" className="block text-sm font-medium text-indigo-100 mb-2">
                Password
              </label>
              <div className="relative">
                <input
                  id="password"
                  name="password"
                  type={showPassword ? "text" : "password"}
                  required
                  value={credentials.password}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 pr-12 border border-white/20 rounded-lg bg-white/5 text-white placeholder-indigo-300 focus:outline-none focus:ring-2 focus:ring-white/30 focus:border-white/40 transition duration-200 text-sm sm:text-base"
                  placeholder="Enter your password"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute inset-y-0 right-0 flex items-center pr-3 hover:bg-white/5 rounded-r-lg transition duration-200"
                >
                  {showPassword ? (
                    <EyeOff className="w-4 h-4 sm:w-5 sm:h-5 text-indigo-300 hover:text-white" />
                  ) : (
                    <Eye className="w-4 h-4 sm:w-5 sm:h-5 text-indigo-300 hover:text-white" />
                  )}
                </button>
              </div>
            </div>

            {/* Error Message */}
            {error && (
              <div className="flex items-center space-x-2 text-red-300 bg-red-500/10 border border-red-500/20 rounded-lg p-3">
                <AlertCircle className="w-4 h-4 sm:w-5 sm:h-5 flex-shrink-0" />
                <span className="text-xs sm:text-sm">{error}</span>
              </div>
            )}

            {/* Demo Credentials Box */}
            {/* <div className="bg-indigo-700/20 border border-indigo-600/30 rounded-lg p-3 sm:p-4">
              <p className="text-xs sm:text-sm text-indigo-100 font-medium mb-2">Demo Credentials:</p>
              <div className="space-y-1">
                <p className="text-xs text-indigo-200">
                  Username: <span className="font-mono bg-white/10 px-2 py-1 rounded">saif</span>
                </p>
                <p className="text-xs text-indigo-200">
                  Password: <span className="font-mono bg-white/10 px-2 py-1 rounded">12345</span>
                </p>
              </div>
            </div> */}

            {/* Submit Button */}
            <button
              type="submit"
              disabled={loading}
              className="w-full flex justify-center items-center px-4 py-3 border border-transparent rounded-lg text-white bg-gradient-to-r from-indigo-600 to-indigo-700 hover:from-indigo-700 hover:to-indigo-800 focus:outline-none focus:ring-2 focus:ring-white/30 focus:ring-offset-2 focus:ring-offset-transparent font-medium transition duration-200 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg text-sm sm:text-base"
            >
              {loading ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 sm:h-5 sm:w-5 border-2 border-white border-t-transparent mr-2"></div>
                  Signing in...
                </>
              ) : (
                <>
                  <LogIn className="w-4 h-4 sm:w-5 sm:h-5 mr-2" />
                  Sign In
                </>
              )}
            </button>
          </form>

          {/* Help Text */}
          <div className="mt-6 text-center">
            <p className="text-xs sm:text-sm text-indigo-300">
              Need help? Contact your system administrator
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center mt-8">
          <p className="text-xs sm:text-sm text-indigo-300">
            © 2025 Trend Tactix. All rights reserved.
          </p>
        </div>
      </div>

      {/* Background decoration - Hidden on small screens */}
      <div className="fixed inset-0 -z-10 hidden sm:block">
        <div className="absolute top-10 left-10 w-24 h-24 bg-white/5 rounded-full blur-xl"></div>
        <div className="absolute bottom-10 right-10 w-32 h-32 bg-indigo-400/10 rounded-full blur-xl"></div>
        <div className="absolute top-1/3 right-1/4 w-16 h-16 bg-white/5 rounded-full blur-xl"></div>
      </div>
    </div>
  );
};

export default Signin;