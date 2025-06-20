// Updated src/config/api.js with better error handling
// Centralized API configuration

// Determine the API base URL
const getApiBaseUrl = () => {
    // Always use localhost in development
    const API_URL = 'http://localhost:5000/api';
    console.log('Using API URL:', API_URL);
    return API_URL;
  };
  
  export const API_BASE_URL = getApiBaseUrl();
  
  // API endpoints
  export const API_ENDPOINTS = {
    health: '/health',
    kpis: '/analytics/kpis',
    notifications: '/notifications',
    products: '/products',
    stock: '/stock',
    sales: '/sales',
    inventory: '/inventory',
  };
  
  // API request wrapper with better error handling
  export const apiRequest = async (endpoint, options = {}) => {
    const url = API_BASE_URL + endpoint;
    
    const config = {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };
  
    try {
      console.log(`🔄 API Request: ${config.method} ${url}`);
      
      // Add timeout to prevent hanging requests
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout
      
      const response = await fetch(url, {
        ...config,
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        const errorData = await response.text();
        throw new Error(`HTTP ${response.status}: ${response.statusText} - ${errorData}`);
      }
      
      const data = await response.json();
      console.log(`✅ API Response: ${url}`, data);
      return data;
    } catch (error) {
      if (error.name === 'AbortError') {
        console.error(`⏱️ API Timeout: ${url}`);
        throw new Error('Request timeout - backend may be slow or unresponsive');
      }
      console.error(`❌ API Error: ${url}`, error);
      throw error;
    }
  };
  
  // Specific API functions with error handling
  export const api = {
    // Health check with timeout
    health: async () => {
      try {
        return await apiRequest(API_ENDPOINTS.health);
      } catch (error) {
        console.error('Health check failed:', error);
        throw new Error('Backend not responding - check if server is running on port 5000');
      }
    },
    
    // KPI endpoints
    getKPIs: async (year = '2025') => {
      try {
        const params = new URLSearchParams({ year });
        return await apiRequest(`${API_ENDPOINTS.kpis}?${params}`);
      } catch (error) {
        console.error('KPI fetch failed:', error);
        throw error;
      }
    },
    
    // Notifications
    getNotifications: async () => {
      try {
        return await apiRequest(API_ENDPOINTS.notifications);
      } catch (error) {
        console.error('Notifications fetch failed:', error);
        throw error;
      }
    },
    
    // Products
    searchProducts: async (query, year = '2025', limit = 50) => {
      try {
        const params = new URLSearchParams({ year, limit });
        return await apiRequest(`${API_ENDPOINTS.products}/search/${query}?${params}`);
      } catch (error) {
        console.error('Product search failed:', error);
        throw error;
      }
    },
    
    // Stock
    getLowStock: async (year = '2025', threshold = 10) => {
      try {
        const params = new URLSearchParams({ year, threshold });
        return await apiRequest(`${API_ENDPOINTS.stock}/low-stock?${params}`);
      } catch (error) {
        console.error('Low stock fetch failed:', error);
        throw error;
      }
    },
    
    // Inventory
    getInventorySummary: async (year = '2025') => {
      try {
        const params = new URLSearchParams({ year });
        return await apiRequest(`${API_ENDPOINTS.inventory}/summary?${params}`);
      } catch (error) {
        console.error('Inventory summary fetch failed:', error);
        throw error;
      }
    },
  };
  
  // Connection test function with detailed error reporting
  export const testConnection = async () => {
    try {
      console.log('Testing connection to backend...');
      const response = await api.health();
      console.log('🔗 Backend connection successful:', response);
      return true;
    } catch (error) {
      console.log('❌ Backend connection failed:', error);
      
      // More specific error handling
      if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
        console.log('💡 Backend may not be running. Check if server is started on port 5000');
      } else if (error.message.includes('CORS')) {
        console.log('💡 CORS error - check backend CORS configuration');
      } else if (error.message.includes('timeout')) {
        console.log('💡 Request timeout - backend is slow or unresponsive');
      }
      
      return false;
    }
  };
  
  export default api;