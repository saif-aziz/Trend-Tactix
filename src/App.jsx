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

import React, { useState } from 'react';
import Sidebar from './Components/Sidebar';
import KPIDashboard from './Components/KPIDashboard';
import SalesDashboard from './Components/SalesDashboard';
import NotificationsPage from './Components/NotificationsPage'; // ✅ New file

export default function App() {
  const [activeItem, setActiveItem] = useState('dashboard');

  return (
    <div className="flex h-screen bg-gray-100 overflow-hidden">
      <Sidebar activeItem={activeItem} setActiveItem={setActiveItem} />
      <div className="flex-1 overflow-auto">
        {activeItem === 'dashboard' && <KPIDashboard />}
        {activeItem === 'analytics' && <SalesDashboard />}
        {activeItem === 'notifications' && <NotificationsPage />} {/* ✅ Updated */}
      </div>
    </div>
  );
}



