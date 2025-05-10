import React from 'react';
import { Calendar, Bell, ChevronDown, Search, BarChart2, AlertCircle } from 'lucide-react';

export default function Header({ activeTab, setActiveTab }) {
  return (
    <header className="bg-white border-b border-gray-200 py-4 px-6 shadow-sm">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <h1 className="text-xl font-bold text-gray-800">Key Perfomance Indicators</h1>
          <div className="hidden md:flex items-center bg-gray-100 rounded-lg px-3 py-1.5">
            <Calendar className="h-4 w-4 text-gray-500 mr-2" />
            <span className="text-sm text-gray-600">May 5, 2025</span>
          </div>
        </div>
        <div className="flex items-center space-x-3">
          <div className="relative hidden md:block">
            <Search className="h-4 w-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
            <input 
              type="text" 
              className="bg-gray-100 rounded-lg pl-9 pr-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 w-48 lg:w-64"
              placeholder="Search..."
            />
          </div>
          <button className="p-2 rounded-lg hover:bg-gray-100 relative">
            <Bell className="h-5 w-5 text-gray-500" />
            <span className="absolute top-0 right-0 h-3 w-3 bg-red-500 rounded-full"></span>
          </button>
          <div className="flex items-center border-l pl-3 ml-3">
            <div className="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center text-white font-medium">
              SA
            </div>
            <button className="flex items-center ml-2 text-sm font-medium text-gray-700 hover:text-gray-900">
              <span className="hidden lg:inline">Saif Aziz</span>
              <ChevronDown className="w-4 h-4 ml-1" />
            </button>
          </div>
        </div>
      </div>
     
    </header>
  );
}
