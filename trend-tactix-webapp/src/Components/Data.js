export const metrics = [
    { title: 'GMROI', value: '5.8', last: '5.5', trend: 'up', category: 'profitability' },
    { title: 'Gross Margin', value: '42%', last: '45%', trend: 'down', category: 'profitability' },
    { title: 'Inventory Turnover', value: '3.2', last: '3.0', trend: 'up', category: 'inventory' },
    { title: 'Weeks of Stock', value: '8', last: '9', trend: 'down', category: 'inventory' },
    { title: 'Markdown', value: '10%', last: '12%', trend: 'up', category: 'pricing' },
    { title: 'Return Rate', value: '5%', last: '4%', trend: 'down', category: 'customer' },
    { title: 'Shrinkage', value: '2%', last: '1.8%', trend: 'down', category: 'operations' },
    { title: 'Avg Monthly Sales', value: '$125K', last: '$120K', trend: 'up', category: 'sales' },
    { title: 'Avg Sell Thru', value: '60%', last: '58%', trend: 'up', category: 'inventory' },
    { title: 'Avg Basket Value', value: '$45', last: '$50', trend: 'down', category: 'sales' },
    { title: 'Avg Invoice Value', value: '$150', last: '$140', trend: 'up', category: 'sales' },
    { title: 'Stock to Sales Ratio', value: '3.5', last: '3.8', trend: 'down', category: 'inventory' },
  ];
  
  export const alerts = [
    { id: 1, title: 'Variants Out Of Stock!!!', text: '22 of your active variants are out of stock.', date: 'Mar 12, 2025 • 07:40 AM', type: 'critical' },
    { id: 2, title: 'Stock Reallocation Needed', text: 'Stock levels are unbalanced across stores.', date: 'Mar 10, 2025 • 06:30 PM', type: 'warning' },
    { id: 3, title: 'New Discount Strategy Suggested', text: 'Competitor sales detected.', date: 'Mar 8, 2025 • 03:15 PM', type: 'info' },
    { id: 4, title: 'Excess Stock Warning', text: 'Inventory levels are higher than required.', date: 'Mar 7, 2025 • 01:45 PM', type: 'notice' },
    { id: 5, title: 'Restock Reminder', text: 'Reorder threshold reached for multiple products.', date: 'Mar 5, 2025 • 09:20 AM', type: 'reminder' },
  ];
  
  export const categories = [
    { name: 'all', label: 'All Metrics' },
    { name: 'profitability', label: 'Profitability' },
    { name: 'inventory', label: 'Inventory' },
    { name: 'sales', label: 'Sales' },
    { name: 'pricing', label: 'Pricing' },
    { name: 'customer', label: 'Customer' },
    { name: 'operations', label: 'Operations' },
  ];
  
  export const shops = [
    { id: 'all', name: 'All Shops' },
    { id: 'shop-a', name: 'Shop A' },
    { id: 'shop-b', name: 'Shop B' },
    { id: 'shop-c', name: 'Shop C' },
    { id: 'shop-d', name: 'Shop D' },
  ];
  
  export const timeframes = [
    { id: 'daily', name: 'Daily' },
    { id: 'weekly', name: 'Weekly' },
    { id: 'monthly', name: 'Monthly' },
    { id: 'quarterly', name: 'Quarterly' },
    { id: 'yearly', name: 'Yearly' },
  ];