// import React from 'react';
// import KPIDashboard      from './Components/KPIDashboard';

// export default function App() {
//   return <KPIDashboard />;
// }

// import React from 'react';
// import SalesDashboard    from './Components/SalesDashboard';
// import KPIDashboard      from './Components/KPIDashboard';
// import NotificationCard from './Components/NotificationCard';

// export default function App() {
//   return (
//     <NotificationCard />
//     // or switch between pages via your router
//   );
// }


//yesterday

// import React, { useState } from 'react';
// import Sidebar from './Components/Sidebar';
// import KPIDashboard from './Components/KPIDashboard';
// import SalesDashboard from './Components/SalesDashboard';
// import NotificationsPage from './Components/NotificationsPage'; // ✅ New file
// import InventoryDashboard from './Components/InventoryDashboard';
// import UserManagement from './Components/UserManagement';

// export default function App() {
//   const [activeItem, setActiveItem] = useState('dashboard');

//   return (
//     <div className="flex h-screen bg-gray-100 overflow-hidden">
//       <Sidebar activeItem={activeItem} setActiveItem={setActiveItem} />
//       <div className="flex-1 overflow-auto">
//         {activeItem === 'dashboard' && <KPIDashboard />}
//         {activeItem === 'analytics' && <SalesDashboard />}
//         {activeItem === 'notifications' && <NotificationsPage />} {/* ✅ Updated */}
//         {activeItem === 'inventory' && <InventoryDashboard />}
//         {activeItem === 'users' && <UserManagement />}
//       </div>
//     </div>
//   );
// }





// Updated App.jsx with Authentication and Role-Based Access Control

import React, { useState, useEffect } from 'react';
import Sidebar from './Components/Sidebar';
import KPIDashboard from './Components/KPIDashboard';
import InventoryDashboard from './Components/InventoryDashboard';
import ModelTraining from './Components/ModelTraining';
import InitialStockDistribution from './Components/InitialStockDistribution';
import NotificationsPage from './Components/NotificationsPage';
import UserManagement from './Components/UserManagement';
import SignIn from './Components/SignIn';
import { RefreshCw, ShieldAlert } from 'lucide-react';

const API_BASE = 'http://localhost:5000/api';

export default function App() {
  // Get initial page from URL hash or localStorage
  const getInitialPage = () => {
    // First check URL hash
    const hash = window.location.hash.replace('#', '');
    if (hash && ['dashboard', 'inventory', 'model-training', 'distribution', 'notifications', 'users'].includes(hash)) {
      return hash;
    }
    // Then check localStorage
    const savedPage = localStorage.getItem('trendTactixCurrentPage');
    if (savedPage && ['dashboard', 'inventory', 'model-training', 'distribution', 'notifications', 'users'].includes(savedPage)) {
      return savedPage;
    }
    return 'dashboard';
  };

  const [activeItem, setActiveItem] = useState(getInitialPage);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [currentUser, setCurrentUser] = useState(null);
  const [userPermissions, setUserPermissions] = useState({});
  const [authToken, setAuthToken] = useState(null);
  const [isCheckingAuth, setIsCheckingAuth] = useState(true);

  // Check for existing authentication on app load
  useEffect(() => {
    checkExistingAuth();
    
    // Listen for hash changes (browser back/forward)
    const handleHashChange = () => {
      const hash = window.location.hash.replace('#', '');
      if (hash && ['dashboard', 'inventory', 'model-training', 'distribution', 'notifications', 'users'].includes(hash)) {
        setActiveItem(hash);
      }
    };
    
    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  // Save current page to localStorage and URL hash when it changes
  useEffect(() => {
    if (activeItem) {
      localStorage.setItem('trendTactixCurrentPage', activeItem);
      window.location.hash = activeItem;
    }
  }, [activeItem]);

  const checkExistingAuth = async () => {
    const savedAuth = localStorage.getItem('trendTactixAuth');
    
    if (savedAuth) {
      try {
        const authData = JSON.parse(savedAuth);
        
        // Validate session with backend
        const response = await fetch(`${API_BASE}/auth/validate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ token: authData.token })
        });
        
        const data = await response.json();
        
        if (data.valid) {
          setIsAuthenticated(true);
          setCurrentUser(data.user);
          setUserPermissions(data.permissions || {});
          setAuthToken(authData.token);
        } else {
          // Session invalid, clear storage
          localStorage.removeItem('trendTactixAuth');
        }
      } catch (err) {
        console.error('Auth validation error:', err);
        // On error, try to use cached data for offline support
        const authData = JSON.parse(savedAuth);
        if (authData.user) {
          setIsAuthenticated(true);
          setCurrentUser(authData.user);
          setUserPermissions(authData.permissions || {});
          setAuthToken(authData.token);
        }
      }
    }
    
    setIsCheckingAuth(false);
  };

  // Handle sign in with real API
  const handleSignIn = async (username, password) => {
    try {
      const response = await fetch(`${API_BASE}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
      });
      
      const data = await response.json();
      
      if (data.success) {
        setIsAuthenticated(true);
        setCurrentUser(data.user);
        setUserPermissions(data.permissions || {});
        setAuthToken(data.token);
        
        // Save to localStorage
        localStorage.setItem('trendTactixAuth', JSON.stringify({
          token: data.token,
          user: data.user,
          permissions: data.permissions,
          timestamp: Date.now()
        }));
        
        return { success: true };
      } else {
        return { success: false, message: data.message };
      }
    } catch (err) {
      console.error('Login error:', err);
      return { success: false, message: 'Failed to connect to server' };
    }
  };

  // Handle sign out
  const handleSignOut = async () => {
    try {
      if (authToken) {
        await fetch(`${API_BASE}/auth/logout`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ token: authToken })
        });
      }
    } catch (err) {
      console.error('Logout error:', err);
    }
    
    setIsAuthenticated(false);
    setCurrentUser(null);
    setUserPermissions({});
    setAuthToken(null);
    setActiveItem('dashboard');
    localStorage.removeItem('trendTactixAuth');
    localStorage.removeItem('trendTactixCurrentPage');
    window.location.hash = 'dashboard';
  };

  // Check if user has permission to view a module
  const hasPermission = (moduleId, type = 'view') => {
    // Admin has all permissions
    if (currentUser?.is_admin) return true;
    
    const perm = userPermissions[moduleId];
    if (!perm) return false;
    
    return type === 'edit' ? perm.edit : perm.view;
  };

  // Handle navigation with permission check
  const handleSetActiveItem = (itemId) => {
    // Admin can access everything
    if (currentUser?.is_admin) {
      setActiveItem(itemId);
      return;
    }
    
    // Check permission for non-admin users
    if (hasPermission(itemId, 'view')) {
      setActiveItem(itemId);
    } else {
      // Redirect to dashboard if no permission
      setActiveItem('dashboard');
    }
  };

  // Show loading while checking auth
  if (isCheckingAuth) {
    return (
      <div className="flex h-screen items-center justify-center bg-gradient-to-br from-indigo-900 to-purple-900">
        <div className="text-center text-white">
          <RefreshCw className="w-12 h-12 animate-spin mx-auto mb-4" />
          <p className="text-lg">Loading Trend Tactix...</p>
        </div>
      </div>
    );
  }

  // If not authenticated, show sign in page
  if (!isAuthenticated) {
    return <SignIn onSignIn={handleSignIn} />;
  }

  // Render content based on active item and permissions
  const renderContent = () => {
    // Check permission for the active item
    if (!currentUser?.is_admin && !hasPermission(activeItem, 'view')) {
      return (
        <div className="flex h-full items-center justify-center">
          <div className="text-center p-8 bg-white rounded-2xl shadow-xl max-w-md">
            <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <ShieldAlert className="w-8 h-8 text-red-600" />
            </div>
            <h2 className="text-xl font-bold text-gray-900 mb-2">Access Denied</h2>
            <p className="text-gray-600 mb-4">
              You don't have permission to access this module.
              Please contact your administrator.
            </p>
            <button
              onClick={() => setActiveItem('dashboard')}
              className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700"
            >
              Go to Dashboard
            </button>
          </div>
        </div>
      );
    }

    switch (activeItem) {
      case 'dashboard':
        return <KPIDashboard currentUser={currentUser} />;
      case 'inventory':
        return <InventoryDashboard />;
      case 'model-training':
        return <ModelTraining />;
      case 'distribution':
        return <InitialStockDistribution />;
      case 'notifications':
        return <NotificationsPage />;
      case 'users':
        // Only admins can access user management
        if (currentUser?.is_admin) {
          return <UserManagement currentUser={currentUser} />;
        }
        return (
          <div className="flex h-full items-center justify-center">
            <div className="text-center p-8 bg-white rounded-2xl shadow-xl max-w-md">
              <ShieldAlert className="w-16 h-16 text-red-500 mx-auto mb-4" />
              <h2 className="text-xl font-bold text-gray-900 mb-2">Admin Access Required</h2>
              <p className="text-gray-600">Only administrators can access user management.</p>
            </div>
          </div>
        );
      default:
        return <KPIDashboard currentUser={currentUser} />;
    }
  };

  // Main application with sidebar and content
  return (
    <div className="flex h-screen bg-gray-100 overflow-hidden">
      <Sidebar 
        activeItem={activeItem} 
        setActiveItem={handleSetActiveItem}
        currentUser={currentUser}
        userPermissions={userPermissions}
        onSignOut={handleSignOut}
      />
      <div className="flex-1 overflow-auto">
        {renderContent()}
      </div>
    </div>
  );
}



