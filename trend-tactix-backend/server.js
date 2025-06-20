// Backend Server with MongoDB Integration
const express = require('express');
const cors = require('cors');
const { MongoClient } = require('mongodb');

const app = express();
const PORT = 5000;

// MongoDB connection
const MONGODB_URI = 'mongodb://localhost:27017';
const DB_NAME = 'Trend-Tactix';

let db;

// Enable CORS
app.use(cors({
  origin: true,
  credentials: true
}));

app.use(express.json());

// Connect to MongoDB
async function connectToDatabase() {
  try {
    const client = new MongoClient(MONGODB_URI, {
      useNewUrlParser: true,
      useUnifiedTopology: true
    });
    await client.connect();
    db = client.db(DB_NAME);
    console.log('✅ Connected to MongoDB:', DB_NAME);
    
    // List collections to verify connection
    const collections = await db.listCollections().toArray();
    console.log('📁 Available collections:', collections.map(c => c.name).join(', '));
  } catch (error) {
    console.error('❌ MongoDB connection error:', error);
    console.log('⚠️  Running without database connection - will use sample data');
  }
}

// Helper function to get stock collection by year
function getStockCollection(year = '2025') {
  const collectionName = `stock${year}`;
  return db ? db.collection(collectionName) : null;
}

// Helper function to get sales collection by year
function getSalesCollection(year = '2025') {
  const collectionName = `${year} sale`;
  return db ? db.collection(collectionName) : null;
}

// Health check endpoint
app.get('/api/health', (req, res) => {
  console.log('Health check requested');
  res.json({ 
    status: 'OK', 
    timestamp: new Date().toISOString(),
    database: db ? 'Connected' : 'Disconnected',
    message: 'Backend is running!'
  });
});

// Get KPI analytics
app.get('/api/analytics/kpis', async (req, res) => {
  try {
    const { year = '2025' } = req.query;
    console.log(`Fetching KPIs for year: ${year}`);
    
    if (!db) {
      // Return sample data if no database connection
      console.log('No database connection, returning sample data');
      return res.json({
        revenue: 2500000,
        totalSales: 15000,
        avgBasketValue: 166.67,
        transactionCount: 15000,
        totalStock: 25000,
        stockValue: 1800000,
        activeProducts: 1200,
        lowStockItems: 45,
        outOfStockItems: 12,
        inventoryTurnover: 3.2,
        avgPrice: 120
      });
    }

    // Get collections
    const stockCollection = getStockCollection(year);
    const salesCollection = getSalesCollection(year);

    if (!stockCollection || !salesCollection) {
      console.log('Collections not found, returning sample data');
      return res.json({
        revenue: 2500000,
        totalSales: 15000,
        avgBasketValue: 166.67,
        transactionCount: 15000,
        totalStock: 25000,
        stockValue: 1800000,
        activeProducts: 1200,
        lowStockItems: 45,
        outOfStockItems: 12,
        inventoryTurnover: 3.2,
        avgPrice: 120
      });
    }

    // Calculate sales metrics from your actual data
    const salesData = await salesCollection.aggregate([
      {
        $addFields: {
          saleAmount: { 
            $multiply: [
              { $ifNull: ["$Quantity", 0] }, 
              { $ifNull: ["$Retail Price", 0] }
            ]
          }
        }
      },
      {
        $group: {
          _id: null,
          totalRevenue: { $sum: "$saleAmount" },
          totalQuantity: { $sum: { $ifNull: ["$Quantity", 0] } },
          transactionCount: { $sum: 1 },
          avgPrice: { $avg: { $ifNull: ["$Retail Price", 0] } }
        }
      }
    ]).toArray();

    // Calculate stock metrics from your actual data
    const stockData = await stockCollection.aggregate([
      {
        $addFields: {
          stockValue: { 
            $multiply: [
              { $ifNull: ["$Quantity", 0] }, 
              { $ifNull: ["$Retail Price", 0] }
            ]
          }
        }
      },
      {
        $group: {
          _id: null,
          totalStock: { $sum: { $ifNull: ["$Quantity", 0] } },
          stockValue: { $sum: "$stockValue" },
          activeProducts: { $sum: 1 },
          lowStockCount: {
            $sum: {
              $cond: [
                { $and: [
                    { $gte: [{ $ifNull: ["$Quantity", 0] }, 1] },
                    { $lt: [{ $ifNull: ["$Quantity", 0] }, 10] }
                ]},
                1, 
                0
              ]
            }
          },
          outOfStockCount: {
            $sum: {
              $cond: [
                { $eq: [{ $ifNull: ["$Quantity", 0] }, 0] },
                1, 
                0
              ]
            }
          }
        }
      }
    ]).toArray();

    const sales = salesData[0] || {};
    const stock = stockData[0] || {};

    // Calculate additional metrics
    const avgBasketValue = sales.transactionCount > 0 ? sales.totalRevenue / sales.transactionCount : 0;
    const inventoryTurnover = stock.stockValue > 0 ? (sales.totalRevenue / stock.stockValue) * 12 : 0;

    const kpis = {
      revenue: sales.totalRevenue || 0,
      totalSales: sales.totalQuantity || 0,
      avgBasketValue: avgBasketValue,
      transactionCount: sales.transactionCount || 0,
      totalStock: stock.totalStock || 0,
      stockValue: stock.stockValue || 0,
      activeProducts: stock.activeProducts || 0,
      lowStockItems: stock.lowStockCount || 0,
      outOfStockItems: stock.outOfStockCount || 0,
      inventoryTurnover: inventoryTurnover,
      avgPrice: sales.avgPrice || 0
    };

    console.log('KPIs calculated from MongoDB:', kpis);
    res.json(kpis);
  } catch (error) {
    console.error('Error calculating KPIs:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get notifications based on real data
app.get('/api/notifications', async (req, res) => {
  try {
    console.log('Fetching notifications...');
    
    if (!db) {
      // Return sample notifications if no database connection
      return res.json([
        {
          id: 1,
          title: 'Backend Connected!',
          text: 'Successfully connected to backend API',
          type: 'info',
          date: new Date().toISOString()
        }
      ]);
    }

    const notifications = [];
    const year = '2025';
    
    // Check for low stock items from your actual data
    const stockCollection = getStockCollection(year);
    if (stockCollection) {
      const lowStockItems = await stockCollection.find({
        Quantity: { $gte: 1, $lt: 10 }
      }).sort({ Quantity: 1 }).limit(5).toArray();
      
      lowStockItems.forEach((item, index) => {
        notifications.push({
          id: `low_stock_${index}`,
          title: 'Low Inventory Alert',
          text: `${item['Product Name'] || 'Unknown Product'} (${item['Product Code']}) has only ${item.Quantity} units remaining`,
          type: 'critical',
          date: new Date().toISOString().split('T')[0],
          read: false
        });
      });

      // Check for out of stock items
      const outOfStockItems = await stockCollection.find({
        Quantity: { $eq: 0 }
      }).limit(3).toArray();
      
      outOfStockItems.forEach((item, index) => {
        notifications.push({
          id: `out_stock_${index}`,
          title: 'Out of Stock Alert',
          text: `${item['Product Name'] || 'Unknown Product'} (${item['Product Code']}) is completely out of stock`,
          type: 'critical',
          date: new Date().toISOString().split('T')[0],
          read: false
        });
      });
    }
    
    // Add a success notification
    notifications.unshift({
      id: 'connected',
      title: 'MongoDB Connected!',
      text: 'Successfully connected to MongoDB and displaying real data',
      type: 'info',
      date: new Date().toISOString().split('T')[0],
      read: false
    });
    
    console.log(`Generated ${notifications.length} notifications`);
    res.json(notifications);
  } catch (error) {
    console.error('Error generating notifications:', error);
    res.status(500).json({ error: error.message });
  }
});

// Error handling
app.use((err, req, res, next) => {
  console.error('Unhandled error:', err);
  res.status(500).json({ error: 'Internal server error' });
});

// 404 handler
app.use((req, res) => {
  console.log('404 - Route not found:', req.url);
  res.status(404).json({ error: 'Route not found' });
});

// Start server
async function startServer() {
  await connectToDatabase();
  app.listen(PORT, () => {
    console.log(`🚀 Server running on port ${PORT}`);
    console.log(`📍 Health check: http://localhost:${PORT}/api/health`);
    console.log(`📊 KPI endpoint: http://localhost:${PORT}/api/analytics/kpis`);
    console.log(`🔔 Notifications: http://localhost:${PORT}/api/notifications`);
  });
}

startServer().catch(console.error);

module.exports = app;