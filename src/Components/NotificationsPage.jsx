// File: src/Components/NotificationsPage.jsx
import React, { useState, useEffect } from 'react';
import {
  Bell, Filter, Search, Check, X,
  ArrowUp, ArrowDown, Calendar, RefreshCw
} from 'lucide-react';
import NotificationCard from './NotificationCard';

const notificationsData = [
    {
      id: 1,
      title: "Low Inventory Alert",
      text: "Product 'Widget XL' inventory is below threshold (5 units remaining). Consider restocking soon.",
      date: "2025-05-07",
      type: "critical",
      read: false
    },
    {
      id: 2,
      title: "Seasonal Stock Warning",
      text: "Summer collection items are showing higher than expected demand. Recommend increasing stock levels by 15%.",
      date: "2025-05-06",
      type: "warning",
      read: false
    },
    {
      id: 3,
      title: "Sales Performance Update",
      text: "Q2 sales performance is tracking 7% above target. Marketing campaigns showing strong ROI.",
      date: "2025-05-05",
      type: "info",
      read: true
    },
    {
      id: 4,
      title: "New Product Launch",
      text: "The 'EcoFriendly Series' will be available next week. Please ensure display areas are prepared.",
      date: "2025-05-04",
      type: "notice",
      read: false
    },
    {
      id: 5,
      title: "Inventory Check Due",
      text: "Monthly inventory verification is scheduled for May 15th. Please complete all sales entries by May 14th.",
      date: "2025-05-03",
      type: "reminder",
      read: true
    },
    {
      id: 6,
      title: "Price Adjustment Required",
      text: "Competitor analysis shows our 'Premium Line' is priced 12% higher than market average. Consider price adjustment.",
      date: "2025-05-02",
      type: "warning",
      read: false
    },
    {
      id: 7,
      title: "Supplier Delay Notice",
      text: "Main textile supplier has reported a 2-week delay for upcoming shipments due to transportation issues.",
      date: "2025-05-01",
      type: "critical",
      read: false
    },
    {
      id: 8,
      title: "Customer Feedback Trends",
      text: "Recent customer surveys show 92% satisfaction with new store layout. Checkout experience received highest ratings.",
      date: "2025-04-30",
      type: "info",
      read: true
    },
    {
      id: 9,
      title: "Staff Training Session",
      text: "Mandatory inventory management training scheduled for May 20th. All floor staff must attend one session.",
      date: "2025-04-29",
      type: "notice",
      read: false
    }
    
  ];

export default function NotificationsPage() {
  const [notifications, setNotifications] = useState(notificationsData);
  const [filteredNotifications, setFilteredNotifications] = useState(notificationsData);
  const [isLoading, setIsLoading] = useState(false);
  const [filterType, setFilterType] = useState('all');
  const [filterReadStatus, setFilterReadStatus] = useState('all');
  const [filterDate, setFilterDate] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [sortDirection, setSortDirection] = useState('newest');

  const notificationCounts = {
    all: notifications.length,
    critical: notifications.filter(n => n.type === 'critical').length,
    warning: notifications.filter(n => n.type === 'warning').length,
    info: notifications.filter(n => n.type === 'info').length,
    notice: notifications.filter(n => n.type === 'notice').length,
    reminder: notifications.filter(n => n.type === 'reminder').length,
    unread: notifications.filter(n => !n.read).length
  };

  useEffect(() => {
    setIsLoading(true);
    let filtered = [...notifications];

    if (filterType !== 'all') filtered = filtered.filter(n => n.type === filterType);
    if (filterReadStatus === 'read') filtered = filtered.filter(n => n.read);
    else if (filterReadStatus === 'unread') filtered = filtered.filter(n => !n.read);

    if (filterDate === 'today') {
      const today = new Date().toISOString().split('T')[0];
      filtered = filtered.filter(n => n.date === today);
    } else if (filterDate === 'week') {
      const weekAgo = new Date();
      weekAgo.setDate(weekAgo.getDate() - 7);
      filtered = filtered.filter(n => new Date(n.date) >= weekAgo);
    } else if (filterDate === 'month') {
      const monthAgo = new Date();
      monthAgo.setMonth(monthAgo.getMonth() - 1);
      filtered = filtered.filter(n => new Date(n.date) >= monthAgo);
    }

    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      filtered = filtered.filter(n =>
        n.title.toLowerCase().includes(q) || n.text.toLowerCase().includes(q)
      );
    }

    filtered.sort((a, b) => {
      const dateA = new Date(a.date), dateB = new Date(b.date);
      return sortDirection === 'newest' ? dateB - dateA : dateA - dateB;
    });

    setTimeout(() => {
      setFilteredNotifications(filtered);
      setIsLoading(false);
    }, 300);
  }, [filterType, filterReadStatus, filterDate, searchQuery, sortDirection, notifications]);

  const markAsRead = (id) => {
    setNotifications(prev =>
      prev.map(n => n.id === id ? { ...n, read: true } : n)
    );
  };

  const markAllAsRead = () => {
    setNotifications(prev => prev.map(n => ({ ...n, read: true })));
  };

  const clearAllNotifications = () => {
    setIsLoading(true);
    setTimeout(() => {
      setNotifications([]);
      setIsLoading(false);
    }, 300);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-6 flex justify-between items-center">
          <h1 className="text-2xl font-bold text-gray-900 flex items-center">
            <Bell className="h-6 w-6 mr-2 text-indigo-600" />
            Notifications
            <span className="ml-2 bg-indigo-100 text-indigo-800 text-xs font-medium px-2.5 py-0.5 rounded-full">
              {notificationCounts.unread} Unread
            </span>
          </h1>
          <div className="flex gap-3">
            <button onClick={markAllAsRead} className="btn">
              <Check className="h-4 w-4 mr-1 text-green-500" />
              Mark All Read
            </button>
            <button onClick={clearAllNotifications} className="btn">
              <X className="h-4 w-4 mr-1 text-red-500" />
              Clear All
            </button>
          </div>
        </div>
      </header>

      {/* Filters */}
      <div className="max-w-7xl mx-auto px-4 py-6 space-y-4 bg-white shadow-sm rounded-lg mb-6">
        <div className="flex flex-col md:flex-row gap-4">
          <div className="relative flex-1">
            <Search className="absolute top-2.5 left-3 text-gray-400 h-5 w-5" />
            <input
              type="text"
              className="w-full pl-10 pr-3 py-2 border rounded-md text-sm"
              placeholder="Search notifications..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>

          <select value={filterType} onChange={(e) => setFilterType(e.target.value)} className="select">
            <option value="all">All Types ({notificationCounts.all})</option>
            <option value="critical">Critical</option>
            <option value="warning">Warning</option>
            <option value="info">Info</option>
            <option value="notice">Notice</option>
            <option value="reminder">Reminder</option>
          </select>

          <select value={filterReadStatus} onChange={(e) => setFilterReadStatus(e.target.value)} className="select">
            <option value="all">All Status</option>
            <option value="read">Read</option>
            <option value="unread">Unread</option>
          </select>

          <select value={filterDate} onChange={(e) => setFilterDate(e.target.value)} className="select">
            <option value="all">All Time</option>
            <option value="today">Today</option>
            <option value="week">Past Week</option>
            <option value="month">Past Month</option>
          </select>

          <button
            onClick={() => setSortDirection(prev => prev === 'newest' ? 'oldest' : 'newest')}
            className="btn"
          >
            {sortDirection === 'newest' ? <ArrowDown className="h-4 w-4 mr-1" /> : <ArrowUp className="h-4 w-4 mr-1" />}
            {sortDirection === 'newest' ? 'Newest First' : 'Oldest First'}
          </button>
        </div>
      </div>

      {/* Notifications List */}
      <div className="max-w-7xl mx-auto px-4">
        {isLoading ? (
          <div className="h-64 flex items-center justify-center">
            <RefreshCw className="animate-spin h-6 w-6 text-indigo-500" />
            <span className="ml-2">Loading...</span>
          </div>
        ) : filteredNotifications.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredNotifications.map(n => (
              <div key={n.id} onClick={() => markAsRead(n.id)} className="relative">
                {!n.read && <div className="absolute -top-2 -right-2 w-3 h-3 bg-indigo-600 rounded-full" />}
                <NotificationCard
                  title={n.title}
                  text={n.text}
                  date={new Date(n.date).toLocaleDateString()}
                  type={n.type}
                />
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center p-8 bg-white shadow rounded-lg">
            <Bell className="mx-auto h-8 w-8 text-gray-300" />
            <p className="text-gray-600 mt-2">No notifications found</p>
          </div>
        )}
      </div>
    </div>
  );
}
