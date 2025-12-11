// Updated Sidebar.jsx with Model Training and AI Stock Distribution pages
import React from 'react';
import { Home, BarChart2, Bell, Users, Settings, LogOut, Package, Brain, Cpu, Zap } from 'lucide-react';

export default function Sidebar({ activeItem, setActiveItem, currentUser, onSignOut }) {
  const menuItems = [
    { id: 'dashboard', icon: Home, label: 'KPI Dashboard' },
    { id: 'analytics', icon: BarChart2, label: 'Sales Analytics' },
    { id: 'inventory', icon: Package, label: 'Inventory Analytics' },
    { id: 'model-training', icon: Cpu, label: 'Model Training', highlight: true }, // Training setup
    { id: 'distribution', icon: Zap, label: 'AI Predictions' }, // Quick predictions
    { id: 'notifications', icon: Bell, label: 'Notifications' },
    { id: 'users', icon: Users, label: 'Manage Team' },
  ];

  const handleSignOut = () => {
    if (window.confirm('Are you sure you want to sign out?')) {
      onSignOut();
    }
  };

  return (
    <aside className="w-20 lg:w-64 bg-gradient-to-b from-indigo-800 to-indigo-900 text-white flex flex-col h-full">
      {/* Logo */}
      <div className="p-4 flex items-center justify-center lg:justify-start">
        <div className="text-3xl font-bold lg:mr-2">TT</div>
        <span className="hidden lg:block text-xl font-semibold">Trend Tactix</span>
      </div>

      {/* User Info */}
      {currentUser && (
        <div className="px-4 py-3 border-b border-indigo-700 hidden lg:block">
          <div className="flex items-center">
            <div className="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center text-white font-medium">
              {currentUser.name?.charAt(0) || 'S'}
            </div>
            <div className="ml-3">
              <p className="text-sm font-medium text-white">{currentUser.name || 'Saif Aziz'}</p>
              <p className="text-xs text-indigo-300">{currentUser.role || 'Admin'}</p>
            </div>
          </div>
        </div>
      )}

      {/* Navigation Menu */}
      <div className="mt-8 px-2 flex-1">
        <div className="hidden lg:block text-xs text-indigo-300 font-medium uppercase tracking-wider mb-2 ml-4">
          Main Menu
        </div>
        <nav className="flex flex-col items-center lg:items-stretch space-y-2">
          {menuItems.map((item) => (
            <button
              key={item.id}
              className={`flex items-center py-3 px-3 lg:px-4 rounded-lg w-full transition-colors ${
                activeItem === item.id 
                  ? 'bg-indigo-700 text-white' 
                  : item.highlight 
                    ? 'text-indigo-100 hover:bg-indigo-700/50 bg-indigo-800/50'
                    : 'text-indigo-200 hover:bg-indigo-700/50'
              }`}
              onClick={() => setActiveItem(item.id)}
            >
              <item.icon className="w-5 h-5 flex-shrink-0" />
              <span className="hidden lg:block ml-3">{item.label}</span>
              {item.id === 'notifications' && (
                <span className="ml-auto bg-red-500 text-white text-xs font-bold rounded-full h-5 w-5 flex items-center justify-center">
                  9
                </span>
              )}
              {item.highlight && activeItem !== item.id && (
                <span className="hidden lg:block ml-auto bg-purple-500 text-white text-xs font-bold rounded px-1.5 py-0.5">
                  Setup
                </span>
              )}
            </button>
          ))}
        </nav>
      </div>

      {/* Sign Out Button */}
      <div className="p-4 border-t border-indigo-700">
        <button 
          onClick={handleSignOut}
          className="flex items-center justify-center lg:justify-start w-full py-2 px-3 rounded-lg text-indigo-200 hover:bg-indigo-700/50 transition-colors"
        >
          <LogOut className="w-5 h-5" />
          <span className="hidden lg:block ml-3">Sign Out</span>
        </button>
      </div>
    </aside>
  );
}