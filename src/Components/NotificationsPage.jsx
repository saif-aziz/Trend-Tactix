import React, { useState, useEffect } from 'react';
import {
  Bell, Filter, Search, Check, X,
  ArrowUp, ArrowDown, Calendar, RefreshCw,
  Eye, AlertTriangle, Package, TrendingUp,
  MapPin, ArrowRight, FileText, CheckCircle
} from 'lucide-react';
import NotificationCard from './NotificationCard';

// Enhanced notifications data with detailed information and action options
const notificationsData = [
  {
    id: 1,
    title: "Low Inventory Alert",
    text: "Product 'Blue Denim Jacket' inventory is below threshold (8 units remaining). Consider restocking soon.",
    date: "2025-05-07",
    type: "critical",
    read: false,
    category: "inventory",
    priority: "high",
    details: {
      product: {
        name: "Blue Denim Jacket",
        sku: "BDJ-001",
        currentStock: 8,
        reorderLevel: 25,
        maxStock: 100,
        avgDailySales: 3.2,
        daysUntilStockout: 2.5
      },
      location: {
        shop: "Shop A",
        section: "Men's Wear - Outerwear"
      },
      analysis: {
        demandTrend: "increasing",
        seasonality: "high",
        profitMargin: 45,
        supplierLeadTime: 7
      },
      actions: [
        {
          type: "warehouse_transfer",
          available: true,
          quantity: 42,
          location: "Central Warehouse",
          timeRequired: "4-6 hours",
          cost: 0
        },
        {
          type: "shop_transfer",
          available: true,
          quantity: 18,
          location: "Shop B",
          timeRequired: "2-3 hours",
          cost: 15
        },
        {
          type: "supplier_order",
          available: true,
          quantity: 100,
          location: "Supplier ABC",
          timeRequired: "7-10 days",
          cost: 1200
        }
      ]
    }
  },
  {
    id: 2,
    title: "Seasonal Stock Warning",
    text: "Summer collection items showing 20% higher demand than forecasted.",
    date: "2025-05-06",
    type: "warning",
    read: false,
    category: "inventory",
    priority: "medium",
    details: {
      product: {
        name: "Summer Collection",
        totalItems: 45,
        affectedItems: 18,
        demandIncrease: 20,
        projectedStockout: "2 weeks"
      },
      location: {
        shop: "All Shops",
        mostAffected: ["Shop C", "Shop A"]
      },
      analysis: {
        salesVelocity: "35% above forecast",
        marketTrend: "trending",
        competitorActivity: "aggressive pricing",
        weatherImpact: "favorable"
      },
      actions: [
        {
          type: "bulk_transfer",
          available: true,
          description: "Transfer from warehouse to high-demand locations",
          estimatedValue: "$45,000",
          timeRequired: "1-2 days"
        },
        {
          type: "emergency_order",
          available: true,
          description: "Place urgent supplier order for trending items",
          estimatedValue: "$78,000",
          timeRequired: "5-7 days"
        }
      ]
    }
  },
  {
    id: 3,
    title: "Excess Stock Alert",
    text: "Winter collection showing slow movement in Shop D. 180 units above optimal level.",
    date: "2025-05-05",
    type: "warning",
    read: true,
    category: "inventory",
    priority: "medium",
    details: {
      product: {
        name: "Winter Collection",
        excessStock: 180,
        currentValue: "$27,000",
        weeksOfSupply: 12,
        markdownRecommendation: "15-25%"
      },
      location: {
        shop: "Shop D",
        section: "Multiple Categories"
      },
      analysis: {
        agingAnalysis: "60% older than 90 days",
        carryingCost: "$135/week",
        opportunityCost: "high",
        seasonalRelevance: "declining"
      },
      actions: [
        {
          type: "inter_shop_transfer",
          available: true,
          description: "Transfer to colder regions (Shop E, Shop F)",
          quantity: 120,
          timeRequired: "3-5 days"
        },
        {
          type: "promotional_pricing",
          available: true,
          description: "Apply 20% markdown to accelerate sales",
          projectedRevenue: "$21,600",
          timeRequired: "immediate"
        }
      ]
    }
  },
  {
    id: 4,
    title: "Stockout Prevention Alert",
    text: "Trending item 'Eco-Friendly Tote Bag' projected to stockout in 3 days.",
    date: "2025-05-04",
    type: "critical",
    read: false,
    category: "inventory",
    priority: "urgent",
    details: {
      product: {
        name: "Eco-Friendly Tote Bag",
        sku: "ETB-205",
        currentStock: 12,
        projectedStockout: 3,
        salesVelocity: 4.5,
        trendingScore: 95
      },
      location: {
        shop: "Shop A",
        category: "Accessories"
      },
      analysis: {
        socialMediaMentions: "+340%",
        competitorStock: "limited",
        marginImpact: "high value item",
        customerSentiment: "very positive"
      },
      actions: [
        {
          type: "urgent_transfer",
          available: true,
          quantity: 25,
          locations: ["Shop B (15)", "Shop C (10)"],
          timeRequired: "same day",
          priority: "urgent"
        },
        {
          type: "express_order",
          available: true,
          quantity: 50,
          supplier: "GreenLine Manufacturing",
          timeRequired: "2-3 days",
          extraCost: 200
        }
      ]
    }
  },
  {
    id: 5,
    title: "Stock Shuffling Suggestion",
    text: "Consider transferring 30 units of 'Men's Running Shoes' from Shop B to Shop D to balance stock levels.",
    date: "2025-05-03",
    type: "info",
    read: false,
    category: "inventory",
    priority: "medium",
    details: {
      product: {
        name: "Men's Running Shoes",
        sku: "MRS-104",
        currentStockShopB: 50,
        currentStockShopD: 15
      },
      location: {
        from: "Shop B",
        to: "Shop D"
      },
      analysis: {
        salesVelocityShopB: 3.0,
        salesVelocityShopD: 4.8,
        imbalanceLevel: "high"
      },
      actions: [
        {
          type: "inter_shop_transfer",
          available: true,
          quantity: 30,
          fromLocation: "Shop B",
          toLocation: "Shop D",
          timeRequired: "3 hours",
          cost: 10
        }
      ]
    }
  },
  {
    id: 6,
    title: "Warehouse Overstock Alert",
    text: "Central Warehouse holding 500 units of 'Kids Jackets', exceeding maximum storage capacity by 50 units.",
    date: "2025-05-02",
    type: "warning",
    read: false,
    category: "inventory",
    priority: "high",
    details: {
      product: {
        name: "Kids Jackets",
        sku: "KJ-220",
        currentStockWarehouse: 500,
        maxCapacity: 450
      },
      location: {
        warehouse: "Central Warehouse"
      },
      analysis: {
        carryingCost: "$200/week",
        suggestedAction: "Redistribute excess stock to shops"
      },
      actions: [
        {
          type: "bulk_transfer",
          available: true,
          description: "Transfer 50 units to Shop E and Shop F",
          quantity: 50,
          toLocations: ["Shop E", "Shop F"],
          timeRequired: "1 day"
        }
      ]
    }
  },
  {
    id: 7,
    title: "Stock Replenishment Completed",
    text: "Restocked 100 units of 'Women's Jackets' to Shop C from warehouse.",
    date: "2025-05-01",
    type: "notice",
    read: true,
    category: "inventory",
    priority: "low",
    details: {
      product: {
        name: "Women's Jackets",
        sku: "WJ-342",
        quantityRestocked: 100
      },
      location: {
        from: "Central Warehouse",
        to: "Shop C"
      },
      analysis: {
        postReplenishmentStock: 120,
        salesVelocity: 3.5
      },
      actions: []
    }
  },
  {
    id: 8,
    title: "Urgent Stock Transfer Required",
    text: "Shop A requires immediate transfer of 40 units of 'Kids Sneakers' to avoid stockout.",
    date: "2025-04-30",
    type: "critical",
    read: false,
    category: "inventory",
    priority: "urgent",
    details: {
      product: {
        name: "Kids Sneakers",
        sku: "KS-309",
        currentStockShopA: 5,
        criticalLevel: 10
      },
      location: {
        from: "Shop B",
        to: "Shop A"
      },
      analysis: {
        salesVelocityShopA: 5.0,
        daysUntilStockout: 1
      },
      actions: [
        {
          type: "urgent_transfer",
          available: true,
          quantity: 40,
          fromLocation: "Shop B",
          toLocation: "Shop A",
          timeRequired: "same day"
        }
      ]
    }
  },
  
  {
    id: 10,
    title: "Projected Stockout Alert",
    text: "Based on sales trends, 'Winter Gloves' expected to stockout in 5 days at Shop F.",
    date: "2025-04-28",
    type: "critical",
    read: false,
    category: "inventory",
    priority: "urgent",
    details: {
      product: {
        name: "Winter Gloves",
        sku: "WG-157",
        currentStock: 20,
        projectedStockoutDays: 5,
        avgDailySales: 4
      },
      location: {
        shop: "Shop F"
      },
      analysis: {
        seasonalDemand: "high",
        supplierLeadTime: 10
      },
      actions: [
        {
          type: "supplier_order",
          available: true,
          quantity: 100,
          location: "Supplier DEF",
          timeRequired: "7-10 days",
          cost: 900
        },
        {
          type: "inter_shop_transfer",
          available: true,
          quantity: 15,
          fromLocation: "Shop E",
          toLocation: "Shop F",
          timeRequired: "1 day"
        }
      ]
    }
  }
];



// Mock data for transfer suggestions
const getTransferSuggestions = (productSku, requiredQuantity) => {
  const suggestions = [
    {
      type: "warehouse",
      location: "Central Warehouse",
      available: Math.floor(Math.random() * 100) + 50,
      distance: "15 km",
      transferTime: "4-6 hours",
      cost: 0,
      reliability: 95
    },
    {
      type: "shop",
      location: "Shop B",
      available: Math.floor(Math.random() * 30) + 10,
      distance: "8 km",
      transferTime: "2-3 hours",
      cost: Math.floor(Math.random() * 50) + 10,
      reliability: 90
    },
    {
      type: "shop",
      location: "Shop C",
      available: Math.floor(Math.random() * 25) + 5,
      distance: "12 km",
      transferTime: "3-4 hours",
      cost: Math.floor(Math.random() * 40) + 15,
      reliability: 88
    }
  ];
  
  return suggestions.filter(s => s.available >= requiredQuantity * 0.5);
};

export default function NotificationsPage() {
  const [notifications, setNotifications] = useState(notificationsData);
  const [filteredNotifications, setFilteredNotifications] = useState(notificationsData);
  const [isLoading, setIsLoading] = useState(false);
  const [filterType, setFilterType] = useState('all');
  const [filterReadStatus, setFilterReadStatus] = useState('all');
  const [filterDate, setFilterDate] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [sortDirection, setSortDirection] = useState('newest');

  // Modal states
  const [selectedNotification, setSelectedNotification] = useState(null);
  const [showDetailModal, setShowDetailModal] = useState(false);
  const [showSTRModal, setShowSTRModal] = useState(false);
  const [selectedAction, setSelectedAction] = useState(null);
  const [strDetails, setSTRDetails] = useState({
    fromLocation: '',
    toLocation: '',
    quantity: '',
    urgency: 'normal',
    notes: ''
  });

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

  const handleViewDetails = (notification) => {
    setSelectedNotification(notification);
    setShowDetailModal(true);
    markAsRead(notification.id);
  };

  const handleCreateSTR = (action, notification) => {
    setSelectedAction(action);
    setSelectedNotification(notification);
    setSTRDetails({
      fromLocation: action.location || '',
      toLocation: notification.details?.location?.shop || '',
      quantity: action.quantity?.toString() || '',
      urgency: notification.priority === 'urgent' ? 'urgent' : 'normal',
      notes: `Auto-generated STR for ${notification.title}`
    });
    setShowSTRModal(true);
  };

  const submitSTR = () => {
    console.log('Submitting STR:', {
      notification: selectedNotification.id,
      action: selectedAction,
      details: strDetails
    });

    setIsLoading(true);
    setTimeout(() => {
      alert('STR submitted successfully to POS system');
      setShowSTRModal(false);
      setIsLoading(false);

      setNotifications(prev =>
        prev.map(n =>
          n.id === selectedNotification.id
            ? { ...n, status: 'addressed', read: true }
            : n
        )
      );
    }, 1000);
  };

  const getPriorityColor = (priority) => {
    switch (priority) {
      case 'urgent': return 'text-red-600 bg-red-100';
      case 'high': return 'text-orange-600 bg-orange-100';
      case 'medium': return 'text-yellow-600 bg-yellow-100';
      default: return 'text-blue-600 bg-blue-100';
    }
  };

  const getActionIcon = (type) => {
    switch (type) {
      case 'warehouse_transfer':
      case 'bulk_transfer':
      case 'urgent_transfer':
        return <Package className="w-4 h-4" />;
      case 'shop_transfer':
      case 'inter_shop_transfer':
        return <MapPin className="w-4 h-4" />;
      case 'supplier_order':
      case 'emergency_order':
      case 'express_order':
        return <TrendingUp className="w-4 h-4" />;
      default:
        return <ArrowRight className="w-4 h-4" />;
    }
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
              <div key={n.id} className="relative">
                {!n.read && <div className="absolute -top-2 -right-2 w-3 h-3 bg-indigo-600 rounded-full" />}
                <div
                  className="bg-white p-4 rounded-lg shadow border-l-4 hover:shadow-lg transition-shadow"
                  style={{
                    borderColor:
                      n.type === 'critical' ? '#ef4444' :
                      n.type === 'warning' ? '#facc15' :
                      n.type === 'info' ? '#3b82f6' :
                      n.type === 'notice' ? '#10b981' :
                      n.type === 'reminder' ? '#8b5cf6' : '#d1d5db'
                  }}
                >
                  <div className="flex justify-between items-start mb-2">
                    <h3 className="text-lg font-semibold text-gray-800">{n.title}</h3>
                    {n.priority && (
                      <span className={`text-xs px-2 py-1 rounded-full font-medium ${getPriorityColor(n.priority)}`}>
                        {n.priority.toUpperCase()}
                      </span>
                    )}
                  </div>
                  <p className="text-sm text-gray-600 mt-1">{n.text}</p>
                  <p className="text-xs text-gray-400 mt-2">{new Date(n.date).toLocaleDateString()}</p>

                  <div className="mt-4 flex gap-2">
                    <button
                      onClick={() => handleViewDetails(n)}
                      className="flex-1 flex items-center justify-center px-3 py-2 bg-blue-600 text-white text-sm rounded-md hover:bg-blue-700 transition-colors"
                    >
                      <Eye className="w-4 h-4 mr-1" />
                      Details
                    </button>
                    {n.details && n.details.actions && n.details.actions.length > 0 && (
                      <button
                        onClick={() => handleViewDetails(n)}
                        className="flex-1 flex items-center justify-center px-3 py-2 bg-green-600 text-white text-sm rounded-md hover:bg-green-700 transition-colors"
                      >
                        <CheckCircle className="w-4 h-4 mr-1" />
                        Actions
                      </button>
                    )}
                  </div>
                </div>
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

      {/* Detailed Notification Modal */}
      {showDetailModal && selectedNotification && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center p-4 z-50 overflow-auto">
          <div className="relative bg-white rounded-lg shadow-xl w-full max-w-lg sm:max-w-2xl md:max-w-4xl max-h-[90vh] overflow-y-auto">
            <button
              onClick={() => setShowDetailModal(false)}
              className="absolute top-3 right-3 text-gray-400 hover:text-gray-600 focus:outline-none"
              aria-label="Close modal"
            >
              <X className="w-6 h-6" />
            </button>

            <div className="px-6 py-4 border-b border-gray-200">
              <h3 className="text-xl font-bold text-gray-900">{selectedNotification.title}</h3>
              <p className="text-sm text-gray-500 mt-1">
                {selectedNotification.details?.location?.shop} • {new Date(selectedNotification.date).toLocaleDateString()}
              </p>
            </div>

            <div className="p-6">
              {/* Product Information */}
              {selectedNotification.details?.product && (
                <div className="mb-6">
                  <h4 className="text-lg font-semibold text-gray-900 mb-3">Product Information</h4>
                  <div className="bg-gray-50 p-4 rounded-lg grid grid-cols-2 md:grid-cols-3 gap-4">
                    {Object.entries(selectedNotification.details.product).map(([key, value]) => (
                      <div key={key}>
                        <span className="text-sm text-gray-500 capitalize">{key.replace(/([A-Z])/g, ' $1').trim()}:</span>
                        <p className="text-sm font-medium text-gray-900">{value}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Analysis */}
              {selectedNotification.details?.analysis && (
                <div className="mb-6">
                  <h4 className="text-lg font-semibold text-gray-900 mb-3">Analysis</h4>
                  <div className="bg-blue-50 p-4 rounded-lg grid grid-cols-1 md:grid-cols-2 gap-4">
                    {Object.entries(selectedNotification.details.analysis).map(([key, value]) => (
                      <div key={key}>
                        <span className="text-sm text-blue-600 capitalize">{key.replace(/([A-Z])/g, ' $1').trim()}:</span>
                        <p className="text-sm font-medium text-gray-900">{value}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Available Actions */}
              {selectedNotification.details?.actions && (
                <div className="mb-6">
                  <h4 className="text-lg font-semibold text-gray-900 mb-3">Available Actions</h4>
                  <div className="space-y-3">
                    {selectedNotification.details.actions.map((action, index) => (
                      <div key={index} className="border border-gray-200 rounded-lg p-4">
                        <div className="flex justify-between items-start">
                          <div className="flex-1">
                            <div className="flex items-center mb-2">
                              {getActionIcon(action.type)}
                              <span className="ml-2 font-medium text-gray-900 capitalize">
                                {action.type.replace(/_/g, ' ')}
                              </span>
                              {action.available && (
                                <span className="ml-2 text-xs bg-green-100 text-green-800 px-2 py-1 rounded-full">
                                  Available
                                </span>
                              )}
                            </div>
                            <p className="text-sm text-gray-600 mb-2">
                              {action.description || `Transfer from ${action.location}`}
                            </p>
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs text-gray-500">
                              {action.quantity && <span>Qty: {action.quantity}</span>}
                              {action.timeRequired && <span>Time: {action.timeRequired}</span>}
                              {action.cost !== undefined && <span>Cost: ${action.cost}</span>}
                              {action.estimatedValue && <span>Value: {action.estimatedValue}</span>}
                            </div>
                          </div>
                          <button
                            onClick={() => handleCreateSTR(action, selectedNotification)}
                            className="ml-4 flex items-center px-3 py-2 bg-green-600 text-white text-sm rounded-md hover:bg-green-700 transition-colors"
                          >
                            <FileText className="w-4 h-4 mr-1" />
                            Create STR
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* STR Creation Modal */}
      {showSTRModal && selectedAction && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center p-4 z-50 overflow-auto">
          <div className="relative bg-white rounded-lg shadow-xl w-full max-w-md sm:max-w-xl md:max-w-2xl max-h-[90vh] overflow-y-auto">
            <button
              onClick={() => setShowSTRModal(false)}
              className="absolute top-3 right-3 text-gray-400 hover:text-gray-600 focus:outline-none"
              aria-label="Close modal"
            >
              <X className="w-6 h-6" />
            </button>

            <div className="px-6 py-4 border-b border-gray-200">
              <h3 className="text-lg font-bold text-gray-900">Create Stock Transfer Request</h3>
              <p className="text-sm text-gray-500">
                {selectedNotification.title} • {selectedAction.type.replace(/_/g, ' ').toUpperCase()}
              </p>
            </div>

            <div className="p-6">
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">From Location</label>
                    <input
                      type="text"
                      value={strDetails.fromLocation}
                      onChange={(e) => setSTRDetails({ ...strDetails, fromLocation: e.target.value })}
                      className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">To Location</label>
                    <input
                      type="text"
                      value={strDetails.toLocation}
                      onChange={(e) => setSTRDetails({ ...strDetails, toLocation: e.target.value })}
                      className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Quantity</label>
                    <input
                      type="number"
                      value={strDetails.quantity}
                      onChange={(e) => setSTRDetails({ ...strDetails, quantity: e.target.value })}
                      className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Urgency Level</label>
                    <select
                      value={strDetails.urgency}
                      onChange={(e) => setSTRDetails({ ...strDetails, urgency: e.target.value })}
                      className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      <option value="normal">Normal</option>
                      <option value="urgent">Urgent</option>
                      <option value="emergency">Emergency</option>
                    </select>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Notes</label>
                  <textarea
                    value={strDetails.notes}
                    onChange={(e) => setSTRDetails({ ...strDetails, notes: e.target.value })}
                    rows={3}
                    className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="Additional notes for the transfer request..."
                  />
                </div>

                {/* Transfer Summary */}
                <div className="bg-blue-50 p-4 rounded-lg">
                  <h4 className="font-medium text-blue-900 mb-2">Transfer Summary</h4>
                  <div className="text-sm text-blue-800 space-y-1">
                    <p>Estimated Time: {selectedAction.timeRequired}</p>
                    {selectedAction.cost !== undefined && <p>Transfer Cost: ${selectedAction.cost}</p>}
                    {selectedAction.extraCost && <p>Extra Cost: ${selectedAction.extraCost}</p>}
                  </div>
                </div>
              </div>
            </div>

            <div className="px-6 py-4 border-t border-gray-200 flex justify-end space-x-3">
              <button
                onClick={() => setShowSTRModal(false)}
                className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                onClick={submitSTR}
                disabled={isLoading}
                className="px-4 py-2 text-sm font-medium text-white bg-blue-600 border border-transparent rounded-md hover:bg-blue-700 disabled:opacity-50"
              >
                {isLoading ? 'Submitting...' : 'Submit STR to POS'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}