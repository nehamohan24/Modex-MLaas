import React, { createContext, useContext, useState, useEffect } from 'react';
import axios from 'axios';
import toast from 'react-hot-toast';
import { getBackendURL } from '../utils/api';

const AuthContext = createContext();

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);

  
  axios.defaults.headers.common['Content-Type'] = 'application/json';

  useEffect(() => {
    const setupBackend = async () => {
      try {
        const backendURL = await getBackendURL();
        axios.defaults.baseURL = backendURL;
      } catch (err) {
        console.error('Failed to set backend URL:', err);
      }
    };
    setupBackend();
  }, []);

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) {
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      setIsAuthenticated(true);
      setUser({
        id: localStorage.getItem('user_id'),
        username: localStorage.getItem('username'),
        domain: localStorage.getItem('domain'),
        subdomain: localStorage.getItem('subdomain')
      });
    }
    setLoading(false);
  }, []);

  const login = async (username, password) => {
    try {
      const response = await axios.post('/login', { username, password });
      const data = response.data;

      if (!data || !data.access_token) {
        throw new Error('Backend did not return access_token');
      }

      
      localStorage.setItem('token', data.access_token);
      localStorage.setItem('user_id', data.user_id || '');
      localStorage.setItem('username', data.username || username);
      localStorage.setItem('domain', data.domain || '');
      localStorage.setItem('subdomain', data.subdomain || '');

      
      axios.defaults.headers.common['Authorization'] = `Bearer ${data.access_token}`;

      
      setUser({
        id: data.user_id || '',
        username: data.username || username,
        domain: data.domain || '',
        subdomain: data.subdomain || ''
      });
      setIsAuthenticated(true);

      toast.success('Login successful!');
      return { success: true };
    } catch (error) {
      console.error('Login error:', error.response?.data || error.message);
      const message = error.response?.data?.error || error.message || 'Login failed';
      toast.error(message);
      return { success: false, error: message };
    }
  };

  const register = async (username, email, password, domain, subdomain) => {
    try {
      const response = await axios.post('/register', { username, email, password, domain, subdomain });
      const data = response.data;

      if (!data || !data.access_token) {
        throw new Error('Backend did not return access_token');
      }

      localStorage.setItem('token', data.access_token);
      localStorage.setItem('user_id', data.user_id || '');
      localStorage.setItem('username', username);
      localStorage.setItem('domain', domain || '');
      localStorage.setItem('subdomain', subdomain || '');

      axios.defaults.headers.common['Authorization'] = `Bearer ${data.access_token}`;

      setUser({
        id: data.user_id || '',
        username,
        domain,
        subdomain
      });
      setIsAuthenticated(true);

      toast.success('Registration successful!');
      return { success: true };
    } catch (error) {
      console.error('Registration error:', error.response?.data || error.message);
      const message = error.response?.data?.error || error.message || 'Registration failed';
      toast.error(message);
      return { success: false, error: message };
    }
  };

  const logout = () => {
    localStorage.clear();
    delete axios.defaults.headers.common['Authorization'];
    setUser(null);
    setIsAuthenticated(false);
    toast.success('Logged out successfully');
  };

  return (
    <AuthContext.Provider value={{ user, isAuthenticated, loading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export default AuthContext;
