import React, { useState, useEffect } from 'react';
import { Search, Filter, Plus, Edit, Trash2, Shield, Eye, Edit3, Users, UserCheck, UserX, Settings } from 'lucide-react';

// Sample user data (simulating POS import)
const sampleUsers = [
  {
    id: 1,
    name: 'Saad Imran',
    email: 'saad.imran@company.com',
    role: 'Store Manager',
    shop: 'Shop A',
    status: 'active',
    permissions: {
      dashboard: 'view',
      analytics: 'edit',
      inventory: 'edit',
      notifications: 'view'
    },
    lastActive: '2025-05-15',
    joinDate: '2023-01-15'
  },
  {
    id: 2,
    name: 'Sarah Khan',
    email: 'sarah.khan@company.com',
    role: 'Sales Associate',
    shop: 'Shop B',
    status: 'active',
    permissions: {
      dashboard: 'view',
      analytics: 'view',
      inventory: 'view',
      notifications: 'view'
    },
    lastActive: '2025-05-14',
    joinDate: '2023-03-22'
  },
  {
    id: 3,
    name: 'Abdul Basit',
    email: 'abdul.basit@company.com',
    role: 'Inventory Manager',
    shop: 'Shop C',
    status: 'active',
    permissions: {
      dashboard: 'view',
      analytics: 'view',
      inventory: 'edit',
      notifications: 'edit'
    },
    lastActive: '2025-05-15',
    joinDate: '2023-02-10'
  },
  {
    id: 4,
    name: 'Abdul Hadi',
    email: 'abdul.hadi@company.com',
    role: 'Assistant Manager',
    shop: 'Shop A',
    status: 'active',
    permissions: {
      dashboard: 'view',
      analytics: 'edit',
      inventory: 'view',
      notifications: 'view'
    },
    lastActive: '2025-05-13',
    joinDate: '2023-06-05'
  },
  {
    id: 5,
    name: 'Moeez Mubashar',
    email: 'moeez.mubashar@company.com',
    role: 'Sales Associate',
    shop: 'Shop D',
    status: 'inactive',
    permissions: {
      dashboard: 'view',
      analytics: 'view',
      inventory: 'view',
      notifications: 'view'
    },
    lastActive: '2025-04-20',
    joinDate: '2023-08-12'
  },
  {
    id: 6,
    name: 'Ali Khan',
    email: 'ali.khan@company.com',
    role: 'Store Manager',
    shop: 'Shop B',
    status: 'active',
    permissions: {
      dashboard: 'edit',
      analytics: 'edit',
      inventory: 'edit',
      notifications: 'edit'
    },
    lastActive: '2025-05-15',
    joinDate: '2023-01-30'
  },
  {
    id: 7,
    name: 'Ahmad Khan',
    email: 'ahmad.khan@company.com',
    role: 'Cashier',
    shop: 'Shop C',
    status: 'active',
    permissions: {
      dashboard: 'view',
      analytics: 'none',
      inventory: 'view',
      notifications: 'view'
    },
    lastActive: '2025-05-14',
    joinDate: '2023-09-18'
  },
  {
    id: 8,
    name: 'Abdullah Ismail',
    email: 'abdullah.ismail@company.com',
    role: 'Visual Merchandiser',
    shop: 'Shop A',
    status: 'active',
    permissions: {
      dashboard: 'view',
      analytics: 'view',
      inventory: 'view',
      notifications: 'view'
    },
    lastActive: '2025-05-15',
    joinDate: '2023-04-25'
  },
  {
    id: 9,
    name: 'Ali Awan',
    email: 'ali.awan@company.com',
    role: 'Security Officer',
    shop: 'Shop D',
    status: 'active',
    permissions: {
      dashboard: 'view',
      analytics: 'none',
      inventory: 'view',
      notifications: 'view'
    },
    lastActive: '2025-05-12',
    joinDate: '2023-07-08'
  },
  {
    id: 10,
    name: 'Shaaf Salman',
    email: 'shaaf.salman@company.com',
    role: 'Regional Manager',
    shop: 'All Shops',
    status: 'active',
    permissions: {
      dashboard: 'edit',
      analytics: 'edit',
      inventory: 'edit',
      notifications: 'edit'
    },
    lastActive: '2025-05-15',
    joinDate: '2022-11-12'
  }
];

const shops = ['All Shops', 'Shop A', 'Shop B', 'Shop C', 'Shop D'];
const roles = ['All Roles', 'Store Manager', 'Assistant Manager', 'Sales Associate', 'Inventory Manager', 'Cashier', 'Visual Merchandiser', 'Security Officer', 'Regional Manager'];
const dashboards = [
  { key: 'dashboard', label: 'KPI Dashboard' },
  { key: 'analytics', label: 'Sales Analytics' },
  { key: 'inventory', label: 'Inventory Management' },
  { key: 'notifications', label: 'Notifications' }
];

export default function UserManagement() {
  const [users, setUsers] = useState(sampleUsers);
  const [filteredUsers, setFilteredUsers] = useState(sampleUsers);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedShop, setSelectedShop] = useState('All Shops');
  const [selectedRole, setSelectedRole] = useState('All Roles');
  const [selectedStatus, setSelectedStatus] = useState('all');
  const [showPermissionModal, setShowPermissionModal] = useState(false);
  const [selectedUser, setSelectedUser] = useState(null);
  const [bulkAction, setBulkAction] = useState('');
  const [selectedUsers, setSelectedUsers] = useState([]);

  // Filter users based on search and filters
  useEffect(() => {
    let filtered = users.filter(user => {
      const matchesSearch = user.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                           user.email.toLowerCase().includes(searchQuery.toLowerCase()) ||
                           user.role.toLowerCase().includes(searchQuery.toLowerCase());
      
      const matchesShop = selectedShop === 'All Shops' || user.shop === selectedShop;
      const matchesRole = selectedRole === 'All Roles' || user.role === selectedRole;
      const matchesStatus = selectedStatus === 'all' || user.status === selectedStatus;
      
      return matchesSearch && matchesShop && matchesRole && matchesStatus;
    });
    
    setFilteredUsers(filtered);
  }, [users, searchQuery, selectedShop, selectedRole, selectedStatus]);

  const handlePermissionChange = (userId, dashboard, permission) => {
    setUsers(users.map(user => 
      user.id === userId 
        ? { 
            ...user, 
            permissions: { 
              ...user.permissions, 
              [dashboard]: permission 
            }
          }
        : user
    ));
  };

  const handleStatusToggle = (userId) => {
    setUsers(users.map(user => 
      user.id === userId 
        ? { ...user, status: user.status === 'active' ? 'inactive' : 'active' }
        : user
    ));
  };

  const handleBulkAction = () => {
    if (!bulkAction || selectedUsers.length === 0) return;
    
    if (bulkAction === 'activate') {
      setUsers(users.map(user => 
        selectedUsers.includes(user.id) ? { ...user, status: 'active' } : user
      ));
    } else if (bulkAction === 'deactivate') {
      setUsers(users.map(user => 
        selectedUsers.includes(user.id) ? { ...user, status: 'inactive' } : user
      ));
    }
    
    setSelectedUsers([]);
    setBulkAction('');
  };

  const handleUserSelect = (userId) => {
    setSelectedUsers(prev => 
      prev.includes(userId) 
        ? prev.filter(id => id !== userId)
        : [...prev, userId]
    );
  };

  const getPermissionIcon = (permission) => {
    switch (permission) {
      case 'edit': return <Edit3 className="w-4 h-4 text-green-600" />;
      case 'view': return <Eye className="w-4 h-4 text-blue-600" />;
      case 'none': return <UserX className="w-4 h-4 text-red-600" />;
      default: return <UserX className="w-4 h-4 text-gray-400" />;
    }
  };

  const getPermissionColor = (permission) => {
    switch (permission) {
      case 'edit': return 'bg-green-100 text-green-800';
      case 'view': return 'bg-blue-100 text-blue-800';
      case 'none': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="flex h-screen overflow-hidden bg-gray-50">
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="bg-white shadow-sm z-10">
          <div className="flex items-center justify-between p-4 border-b border-gray-200">
            <div className="flex items-center">
              <Users className="w-6 h-6 mr-2 text-indigo-600" />
              <h2 className="text-2xl font-bold text-gray-800">User Management</h2>
              <div className="ml-4 text-sm text-gray-500">
                {filteredUsers.length} users • {filteredUsers.filter(u => u.status === 'active').length} active
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <button className="inline-flex items-center px-3 py-2 border border-green-600 text-sm leading-4 font-medium rounded-md text-green-600 bg-white hover:bg-green-50">
                <Plus className="h-4 w-4 mr-2" />
                Import from POS
              </button>
              <button className="inline-flex items-center px-3 py-2 border border-gray-300 text-sm leading-4 font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50">
                <Plus className="h-4 w-4 mr-2" />
                Add User
              </button>
            </div>
          </div>
          
          {/* Filters and Search */}
          <div className="p-4 bg-gray-50 border-b border-gray-200">
            <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
              <div className="md:col-span-2">
                <div className="relative">
                  <Search className="absolute left-3 top-2.5 h-4 w-4 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Search users..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full pl-10 pr-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                  />
                </div>
              </div>
              
              <select
                value={selectedShop}
                onChange={(e) => setSelectedShop(e.target.value)}
                className="border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
              >
                {shops.map(shop => (
                  <option key={shop} value={shop}>{shop}</option>
                ))}
              </select>
              
              <select
                value={selectedRole}
                onChange={(e) => setSelectedRole(e.target.value)}
                className="border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
              >
                {roles.map(role => (
                  <option key={role} value={role}>{role}</option>
                ))}
              </select>
              
              <select
                value={selectedStatus}
                onChange={(e) => setSelectedStatus(e.target.value)}
                className="border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
              >
                <option value="all">All Status</option>
                <option value="active">Active</option>
                <option value="inactive">Inactive</option>
              </select>
            </div>
            
            {/* Bulk Actions */}
            {selectedUsers.length > 0 && (
              <div className="mt-4 flex items-center justify-between bg-blue-50 p-3 rounded-lg">
                <span className="text-sm text-blue-800">
                  {selectedUsers.length} user(s) selected
                </span>
                <div className="flex items-center space-x-3">
                  <select
                    value={bulkAction}
                    onChange={(e) => setBulkAction(e.target.value)}
                    className="border border-blue-300 rounded-md px-3 py-1 text-sm"
                  >
                    <option value="">Select action</option>
                    <option value="activate">Activate</option>
                    <option value="deactivate">Deactivate</option>
                  </select>
                  <button
                    onClick={handleBulkAction}
                    className="px-3 py-1 bg-blue-600 text-white text-sm rounded-md hover:bg-blue-700"
                  >
                    Apply
                  </button>
                </div>
              </div>
            )}
          </div>
        </header>

        {/* User Table */}
        <div className="flex-1 overflow-y-auto p-6">
          <div className="bg-white shadow rounded-lg overflow-hidden">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="w-4 p-3">
                    <input
                      type="checkbox"
                      checked={selectedUsers.length === filteredUsers.length}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSelectedUsers(filteredUsers.map(u => u.id));
                        } else {
                          setSelectedUsers([]);
                        }
                      }}
                      className="rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                    />
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">User</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Role & Shop</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Permissions</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Last Active</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {filteredUsers.map(user => (
                  <tr key={user.id} className="hover:bg-gray-50">
                    <td className="p-3">
                      <input
                        type="checkbox"
                        checked={selectedUsers.includes(user.id)}
                        onChange={() => handleUserSelect(user.id)}
                        className="rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                      />
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <div className="flex-shrink-0 h-10 w-10">
                          <div className="h-10 w-10 rounded-full bg-indigo-500 flex items-center justify-center text-white font-medium">
                            {user.name.charAt(0)}
                          </div>
                        </div>
                        <div className="ml-4">
                          <div className="text-sm font-medium text-gray-900">{user.name}</div>
                          <div className="text-sm text-gray-500">{user.email}</div>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm text-gray-900">{user.role}</div>
                      <div className="text-sm text-gray-500">{user.shop}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex space-x-1">
                        {dashboards.map(dashboard => (
                          <div key={dashboard.key} className="flex items-center">
                            {getPermissionIcon(user.permissions[dashboard.key])}
                          </div>
                        ))}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                        user.status === 'active' 
                          ? 'bg-green-100 text-green-800' 
                          : 'bg-red-100 text-red-800'
                      }`}>
                        {user.status === 'active' ? (
                          <><UserCheck className="w-3 h-3 mr-1" /> Active</>
                        ) : (
                          <><UserX className="w-3 h-3 mr-1" /> Inactive</>
                        )}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {user.lastActive}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                      <div className="flex space-x-2">
                        <button
                          onClick={() => {
                            setSelectedUser(user);
                            setShowPermissionModal(true);
                          }}
                          className="text-indigo-600 hover:text-indigo-900"
                        >
                          <Shield className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => handleStatusToggle(user.id)}
                          className={user.status === 'active' ? 'text-red-600 hover:text-red-900' : 'text-green-600 hover:text-green-900'}
                        >
                          {user.status === 'active' ? <UserX className="w-4 h-4" /> : <UserCheck className="w-4 h-4" />}
                        </button>
                        <button className="text-gray-600 hover:text-gray-900">
                          <Edit className="w-4 h-4" />
                        </button>
                        <button className="text-red-600 hover:text-red-900">
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Permission Modal */}
        {showPermissionModal && selectedUser && (
          <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center p-4 z-50">
            <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-96 overflow-y-auto">
              <div className="px-6 py-4 border-b border-gray-200">
                <h3 className="text-lg font-medium text-gray-900">
                  Manage Permissions for {selectedUser.name}
                </h3>
                <p className="text-sm text-gray-500 mt-1">{selectedUser.email}</p>
              </div>
              <div className="p-6">
                <div className="space-y-4">
                  {dashboards.map(dashboard => (
                    <div key={dashboard.key} className="flex items-center justify-between">
                      <span className="text-sm font-medium text-gray-700">{dashboard.label}</span>
                      <div className="flex space-x-2">
                        {['none', 'view', 'edit'].map(permission => (
                          <button
                            key={permission}
                            onClick={() => handlePermissionChange(selectedUser.id, dashboard.key, permission)}
                            className={`px-3 py-1 text-xs font-medium rounded-full ${
                              selectedUser.permissions[dashboard.key] === permission
                                ? getPermissionColor(permission)
                                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                            }`}
                          >
                            {permission.charAt(0).toUpperCase() + permission.slice(1)}
                          </button>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              <div className="px-6 py-4 border-t border-gray-200 flex justify-end space-x-3">
                <button
                  onClick={() => setShowPermissionModal(false)}
                  className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50"
                >
                  Cancel
                </button>
                <button
                  onClick={() => {
                    setShowPermissionModal(false);
                    setSelectedUser(null);
                  }}
                  className="px-4 py-2 text-sm font-medium text-white bg-indigo-600 border border-transparent rounded-md hover:bg-indigo-700"
                >
                  Save Changes
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}