"""
User Management Database Module
Handles user CRUD, authentication, and role-based permissions
"""
import sqlite3
import os
import hashlib
import secrets
from datetime import datetime
import json


class UserManagementDB:
    """SQLite database manager for user management and RBAC"""
    
    # Define all available modules/screens in the application
    AVAILABLE_MODULES = [
        {'id': 'dashboard', 'name': 'KPI Dashboard', 'description': 'Overview metrics and KPIs'},
        {'id': 'inventory', 'name': 'Sales & Inventory Analytics', 'description': 'Historical data analysis'},
        {'id': 'model-training', 'name': 'Model Training', 'description': 'Train forecasting models'},
        {'id': 'distribution', 'name': 'AI Predictions', 'description': 'Generate demand forecasts'},
        {'id': 'notifications', 'name': 'Notifications', 'description': 'Alerts and updates'},
        {'id': 'users', 'name': 'Manage Team', 'description': 'User administration (Admin only)'}
    ]
    
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), 'data', 'users.db')
        
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_database()
    
    def _get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_database(self):
        """Initialize database tables"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                full_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                designation TEXT,
                status TEXT DEFAULT 'active',
                is_admin INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_login TEXT,
                created_by INTEGER,
                FOREIGN KEY (created_by) REFERENCES users(id)
            )
        ''')
        
        # User permissions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_permissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                module_id TEXT NOT NULL,
                can_view INTEGER DEFAULT 0,
                can_edit INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                UNIQUE(user_id, module_id)
            )
        ''')
        
        # Activity log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                action TEXT NOT NULL,
                details TEXT,
                ip_address TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # Sessions table for token management
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                is_active INTEGER DEFAULT 1,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        ''')
        
        conn.commit()
        
        # Create default admin if no users exist
        cursor.execute('SELECT COUNT(*) FROM users')
        if cursor.fetchone()[0] == 0:
            self._create_default_admin(cursor)
            conn.commit()
        
        conn.close()
    
    def _create_default_admin(self, cursor):
        """Create default admin user"""
        salt = secrets.token_hex(32)
        password_hash = self._hash_password('admin123', salt)
        now = datetime.now().isoformat()
        
        cursor.execute('''
            INSERT INTO users (username, email, full_name, password_hash, salt, role, 
                             designation, status, is_admin, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', ('admin', 'admin@trendtactix.com', 'System Administrator', 
              password_hash, salt, 'admin', 'Administrator', 'active', 1, now, now))
        
        admin_id = cursor.lastrowid
        
        # Grant all permissions to admin
        for module in self.AVAILABLE_MODULES:
            cursor.execute('''
                INSERT INTO user_permissions (user_id, module_id, can_view, can_edit, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (admin_id, module['id'], 1, 1, now, now))
        
        print(f"✅ Default admin user created: username='admin', password='admin123'")
    
    def _hash_password(self, password, salt):
        """Hash password with salt using SHA-256"""
        return hashlib.sha256((password + salt).encode()).hexdigest()
    
    def _verify_password(self, password, salt, password_hash):
        """Verify password against hash"""
        return self._hash_password(password, salt) == password_hash
    
    # ============ USER CRUD OPERATIONS ============
    
    def create_user(self, username, email, full_name, password, role='user', 
                    designation=None, is_admin=False, created_by=None):
        """Create a new user"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            salt = secrets.token_hex(32)
            password_hash = self._hash_password(password, salt)
            now = datetime.now().isoformat()
            
            cursor.execute('''
                INSERT INTO users (username, email, full_name, password_hash, salt, role,
                                 designation, status, is_admin, created_at, updated_at, created_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (username, email, full_name, password_hash, salt, role,
                  designation, 'active', 1 if is_admin else 0, now, now, created_by))
            
            user_id = cursor.lastrowid
            
            # Create default permissions (view only for basic modules)
            default_modules = ['dashboard', 'inventory', 'notifications']
            for module_id in default_modules:
                cursor.execute('''
                    INSERT INTO user_permissions (user_id, module_id, can_view, can_edit, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (user_id, module_id, 1, 0, now, now))
            
            conn.commit()
            
            # Log activity
            self.log_activity(created_by, 'CREATE_USER', f'Created user: {username}')
            
            return {'success': True, 'user_id': user_id, 'message': 'User created successfully'}
            
        except sqlite3.IntegrityError as e:
            if 'username' in str(e):
                return {'success': False, 'message': 'Username already exists'}
            elif 'email' in str(e):
                return {'success': False, 'message': 'Email already exists'}
            return {'success': False, 'message': str(e)}
        finally:
            conn.close()
    
    def get_user(self, user_id):
        """Get user by ID"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, username, email, full_name, role, designation, status, 
                   is_admin, created_at, updated_at, last_login
            FROM users WHERE id = ?
        ''', (user_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
    
    def get_user_by_username(self, username):
        """Get user by username"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
    
    def get_all_users(self):
        """Get all users"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, username, email, full_name, role, designation, status, 
                   is_admin, created_at, updated_at, last_login
            FROM users ORDER BY created_at DESC
        ''')
        
        users = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return users
    
    def update_user(self, user_id, updates, updated_by=None):
        """Update user details"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        allowed_fields = ['email', 'full_name', 'role', 'designation', 'status', 'is_admin']
        set_clauses = []
        values = []
        
        for field, value in updates.items():
            if field in allowed_fields:
                set_clauses.append(f'{field} = ?')
                values.append(value)
        
        if not set_clauses:
            conn.close()
            return {'success': False, 'message': 'No valid fields to update'}
        
        set_clauses.append('updated_at = ?')
        values.append(datetime.now().isoformat())
        values.append(user_id)
        
        cursor.execute(f'''
            UPDATE users SET {', '.join(set_clauses)} WHERE id = ?
        ''', values)
        
        conn.commit()
        conn.close()
        
        self.log_activity(updated_by, 'UPDATE_USER', f'Updated user ID: {user_id}')
        
        return {'success': True, 'message': 'User updated successfully'}
    
    def update_password(self, user_id, new_password, updated_by=None):
        """Update user password"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        salt = secrets.token_hex(32)
        password_hash = self._hash_password(new_password, salt)
        now = datetime.now().isoformat()
        
        cursor.execute('''
            UPDATE users SET password_hash = ?, salt = ?, updated_at = ? WHERE id = ?
        ''', (password_hash, salt, now, user_id))
        
        conn.commit()
        conn.close()
        
        self.log_activity(updated_by, 'UPDATE_PASSWORD', f'Password updated for user ID: {user_id}')
        
        return {'success': True, 'message': 'Password updated successfully'}
    
    def delete_user(self, user_id, deleted_by=None):
        """Delete a user (soft delete - set status to deleted)"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Don't allow deleting the last admin
        cursor.execute('SELECT COUNT(*) FROM users WHERE is_admin = 1 AND status = "active"')
        admin_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT is_admin FROM users WHERE id = ?', (user_id,))
        user = cursor.fetchone()
        
        if user and user['is_admin'] and admin_count <= 1:
            conn.close()
            return {'success': False, 'message': 'Cannot delete the last admin user'}
        
        cursor.execute('''
            UPDATE users SET status = 'deleted', updated_at = ? WHERE id = ?
        ''', (datetime.now().isoformat(), user_id))
        
        conn.commit()
        conn.close()
        
        self.log_activity(deleted_by, 'DELETE_USER', f'Deleted user ID: {user_id}')
        
        return {'success': True, 'message': 'User deleted successfully'}
    
    # ============ AUTHENTICATION ============
    
    def authenticate(self, username, password):
        """Authenticate user and return session token"""
        user = self.get_user_by_username(username)
        
        if not user:
            return {'success': False, 'message': 'Invalid username or password'}
        
        if user['status'] != 'active':
            return {'success': False, 'message': 'Account is inactive'}
        
        if not self._verify_password(password, user['salt'], user['password_hash']):
            self.log_activity(user['id'], 'LOGIN_FAILED', 'Invalid password')
            return {'success': False, 'message': 'Invalid username or password'}
        
        # Generate session token
        token = secrets.token_urlsafe(32)
        now = datetime.now()
        expires = datetime(now.year, now.month, now.day + 7)  # 7 day expiry
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Update last login
        cursor.execute('UPDATE users SET last_login = ? WHERE id = ?', 
                      (now.isoformat(), user['id']))
        
        # Create session
        cursor.execute('''
            INSERT INTO sessions (user_id, token, created_at, expires_at)
            VALUES (?, ?, ?, ?)
        ''', (user['id'], token, now.isoformat(), expires.isoformat()))
        
        conn.commit()
        conn.close()
        
        # Get user permissions
        permissions = self.get_user_permissions(user['id'])
        
        self.log_activity(user['id'], 'LOGIN_SUCCESS', 'User logged in')
        
        return {
            'success': True,
            'token': token,
            'user': {
                'id': user['id'],
                'username': user['username'],
                'email': user['email'],
                'full_name': user['full_name'],
                'role': user['role'],
                'designation': user['designation'],
                'is_admin': bool(user['is_admin'])
            },
            'permissions': permissions
        }
    
    def validate_session(self, token):
        """Validate session token and return user"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT s.*, u.id as user_id, u.username, u.email, u.full_name, 
                   u.role, u.designation, u.is_admin, u.status
            FROM sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.token = ? AND s.is_active = 1
        ''', (token,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return {'valid': False, 'message': 'Invalid session'}
        
        if row['status'] != 'active':
            return {'valid': False, 'message': 'Account is inactive'}
        
        # Check expiry
        expires = datetime.fromisoformat(row['expires_at'])
        if datetime.now() > expires:
            return {'valid': False, 'message': 'Session expired'}
        
        permissions = self.get_user_permissions(row['user_id'])
        
        return {
            'valid': True,
            'user': {
                'id': row['user_id'],
                'username': row['username'],
                'email': row['email'],
                'full_name': row['full_name'],
                'role': row['role'],
                'designation': row['designation'],
                'is_admin': bool(row['is_admin'])
            },
            'permissions': permissions
        }
    
    def logout(self, token):
        """Invalidate session token"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('UPDATE sessions SET is_active = 0 WHERE token = ?', (token,))
        conn.commit()
        conn.close()
        
        return {'success': True, 'message': 'Logged out successfully'}
    
    # ============ PERMISSIONS ============
    
    def get_user_permissions(self, user_id):
        """Get all permissions for a user"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Check if user is admin (admins have all permissions)
        cursor.execute('SELECT is_admin FROM users WHERE id = ?', (user_id,))
        user = cursor.fetchone()
        
        if user and user['is_admin']:
            # Admin gets all permissions
            permissions = {}
            for module in self.AVAILABLE_MODULES:
                permissions[module['id']] = {'view': True, 'edit': True}
            conn.close()
            return permissions
        
        cursor.execute('''
            SELECT module_id, can_view, can_edit
            FROM user_permissions WHERE user_id = ?
        ''', (user_id,))
        
        permissions = {}
        for row in cursor.fetchall():
            permissions[row['module_id']] = {
                'view': bool(row['can_view']),
                'edit': bool(row['can_edit'])
            }
        
        conn.close()
        return permissions
    
    def set_user_permissions(self, user_id, permissions, updated_by=None):
        """Set permissions for a user"""
        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        
        for module_id, rights in permissions.items():
            can_view = 1 if rights.get('view', False) else 0
            can_edit = 1 if rights.get('edit', False) else 0
            
            # Upsert permission
            cursor.execute('''
                INSERT INTO user_permissions (user_id, module_id, can_view, can_edit, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, module_id) DO UPDATE SET
                    can_view = excluded.can_view,
                    can_edit = excluded.can_edit,
                    updated_at = excluded.updated_at
            ''', (user_id, module_id, can_view, can_edit, now, now))
        
        conn.commit()
        conn.close()
        
        self.log_activity(updated_by, 'UPDATE_PERMISSIONS', f'Permissions updated for user ID: {user_id}')
        
        return {'success': True, 'message': 'Permissions updated successfully'}
    
    def get_available_modules(self):
        """Get list of available modules"""
        return self.AVAILABLE_MODULES
    
    # ============ ACTIVITY LOG ============
    
    def log_activity(self, user_id, action, details=None, ip_address=None):
        """Log user activity"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO activity_log (user_id, action, details, ip_address, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, action, details, ip_address, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def get_activity_log(self, user_id=None, limit=50):
        """Get activity log"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if user_id:
            cursor.execute('''
                SELECT al.*, u.username, u.full_name
                FROM activity_log al
                LEFT JOIN users u ON al.user_id = u.id
                WHERE al.user_id = ?
                ORDER BY al.timestamp DESC
                LIMIT ?
            ''', (user_id, limit))
        else:
            cursor.execute('''
                SELECT al.*, u.username, u.full_name
                FROM activity_log al
                LEFT JOIN users u ON al.user_id = u.id
                ORDER BY al.timestamp DESC
                LIMIT ?
            ''', (limit,))
        
        logs = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return logs


# Initialize singleton instance
user_db = UserManagementDB()



