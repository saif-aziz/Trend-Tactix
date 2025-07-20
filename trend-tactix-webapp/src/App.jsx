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





//new code

// Updated App.jsx with Initial Stock Distribution page
import React, { useState, useEffect } from 'react';
import Sidebar from './Components/Sidebar';
import KPIDashboard from './Components/KPIDashboard';
import SalesDashboard from './Components/SalesDashboard';
import InventoryDashboard from './Components/InventoryDashboard';
import InitialStockDistribution from './Components/InitialStockDistribution';
import NotificationsPage from './Components/NotificationsPage';
import UserManagement from './Components/UserManagement';
import SignIn from './Components/SignIn';

export default function App() {
  const [activeItem, setActiveItem] = useState('dashboard');
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [currentUser, setCurrentUser] = useState(null);

  // Check for existing authentication on app load
  useEffect(() => {
    const savedAuthState = localStorage.getItem('trendTactixAuth');
    if (savedAuthState) {
      const authData = JSON.parse(savedAuthState);
      setIsAuthenticated(true);
      setCurrentUser(authData.user);
    }
  }, []);

  // Handle sign in
  const handleSignIn = () => {
    const userData = {
      username: 'saif',
      name: 'Saif Aziz',
      role: 'Admin',
      email: 'saif@trendtactix.com'
    };
    
    setIsAuthenticated(true);
    setCurrentUser(userData);
    
    // Save authentication state to localStorage
    localStorage.setItem('trendTactixAuth', JSON.stringify({
      user: userData,
      timestamp: Date.now()
    }));
  };

  // Handle sign out
  const handleSignOut = () => {
    setIsAuthenticated(false);
    setCurrentUser(null);
    setActiveItem('dashboard'); // Reset to dashboard
    
    // Clear authentication state from localStorage
    localStorage.removeItem('trendTactixAuth');
  };

  // If not authenticated, show sign in page
  if (!isAuthenticated) {
    return <SignIn onSignIn={handleSignIn} />;
  }

  // Main application with sidebar and content
  return (
    <div className="flex h-screen bg-gray-100 overflow-hidden">
      <Sidebar 
        activeItem={activeItem} 
        setActiveItem={setActiveItem}
        currentUser={currentUser}
        onSignOut={handleSignOut}
      />
      <div className="flex-1 overflow-auto">
        {activeItem === 'dashboard' && <KPIDashboard />}
        {activeItem === 'analytics' && <SalesDashboard />}
        {activeItem === 'inventory' && <InventoryDashboard />}
        {activeItem === 'distribution' && <InitialStockDistribution />}
        {activeItem === 'notifications' && <NotificationsPage />}
        {activeItem === 'users' && <UserManagement />}
        {activeItem === 'settings' && (
          <div className="p-6">
            <h1 className="text-2xl font-bold text-gray-900">Settings</h1>
            <p className="text-gray-600 mt-2">Settings page coming soon...</p>
          </div>
        )}
      </div>
    </div>
  );
}



