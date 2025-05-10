import React from 'react';
import { Home, BarChart2, Bell, Users, Settings, LogOut } from 'lucide-react';

export default function Sidebar({ activeItem, setActiveItem }) {
  const menuItems = [
    { id: 'dashboard', icon: Home, label: 'Dashboard' },
    { id: 'analytics', icon: BarChart2, label: 'Analytics' },
    { id: 'notifications', icon: Bell, label: 'Notifications' },
    { id: 'users', icon: Users, label: 'Team' },
    { id: 'settings', icon: Settings, label: 'Settings' },
  ];

  return (
    <aside className="w-20 lg:w-64 bg-gradient-to-b from-indigo-800 to-indigo-900 text-white flex flex-col h-full #070626">
      <div className="p-4 flex items-center justify-center lg:justify-start">
        <div className="text-3xl font-bold lg:mr-2">TT</div>
        <span className="hidden lg:block text-xl font-semibold">Trend Tactix</span>
      </div>
      <div className="mt-8 px-2">
        <div className="hidden lg:block text-xs text-indigo-300 font-medium uppercase tracking-wider mb-2 ml-4">
          Main Menu
        </div>
        <nav className="flex flex-col items-center lg:items-stretch space-y-2">
          {menuItems.map((item) => (
            <button
              key={item.id}
              className={`flex items-center py-3 px-3 lg:px-4 rounded-lg w-full transition-colors ${
                activeItem === item.id ? 'bg-indigo-700 text-white' : 'text-indigo-200 hover:bg-indigo-700/50'
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
            </button>
          ))}
        </nav>
      </div>
      <div className="mt-auto p-4">
        <button className="flex items-center justify-center lg:justify-start w-full py-2 px-3 rounded-lg text-indigo-200 hover:bg-indigo-700/50 transition-colors">
          <LogOut className="w-5 h-5" />
          <span className="hidden lg:block ml-3">Log Out</span>
        </button>
      </div>
    </aside>
  );
}
