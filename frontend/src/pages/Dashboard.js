import React, { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { Link } from 'react-router-dom';
import { getBackendURL } from '../utils/api';
import { 
  Database, 
  Brain, 
  Shield, 
  TrendingUp, 
  Users, 
  BarChart3,
  Activity,
  Zap
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

const Dashboard = () => {
  const { user } = useAuth();
  const [stats, setStats] = useState({
    user_datasets: 0,
    contributed_models: 0,
    total_datasets: 0,
    total_models: 0
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      const token = localStorage.getItem('token');
      const backendURL = await getBackendURL();
      const response = await fetch(`${backendURL}/stats`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      const data = await response.json();
      setStats(data);
    } catch (error) {
      console.error('Failed to fetch stats:', error);
    } finally {
      setLoading(false);
    }
  };

  const performanceData = [
    { name: 'Week 1', accuracy: 85, models: 12 },
    { name: 'Week 2', accuracy: 88, models: 18 },
    { name: 'Week 3', accuracy: 91, models: 25 },
    { name: 'Week 4', accuracy: 94, models: 32 },
  ];

  const domainData = [
    { name: 'Healthcare', value: 35, color: '#3B82F6' },
    { name: 'Finance', value: 25, color: '#10B981' },
    { name: 'Technology', value: 20, color: '#F59E0B' },
    { name: 'Retail', value: 15, color: '#EF4444' },
    { name: 'Education', value: 5, color: '#8B5CF6' },
  ];

  const StatCard = ({ title, value, icon: Icon, color, trend }) => (
    <div className="card hover:shadow-md transition-shadow duration-200">
      <div className="flex items-center">
        <div className={`p-3 rounded-lg ${color}`}>
          <Icon className="h-6 w-6 text-white" />
        </div>
        <div className="ml-4">
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="text-2xl font-semibold text-gray-900">{value}</p>
          {trend && (
            <p className="text-sm text-green-600 flex items-center">
              <TrendingUp className="h-4 w-4 mr-1" />
              {trend}
            </p>
          )}
        </div>
      </div>
    </div>
  );

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">
          Welcome back, {user?.username}!
        </h1>
        <p className="mt-2 text-gray-600">
          Monitor your collaborative ML training progress and model performance.
        </p>
        {user?.domain && (
          <div className="mt-4 inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-primary-100 text-primary-800">
            <Shield className="h-4 w-4 mr-2" />
            {user.domain} - {user.subdomain}
          </div>
        )}
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <StatCard
          title="Your Datasets"
          value={stats.user_datasets}
          icon={Database}
          color="bg-blue-500"
          trend="+12% this month"
        />
        <StatCard
          title="Models Contributed"
          value={stats.contributed_models}
          icon={Brain}
          color="bg-green-500"
          trend="+8% this month"
        />
        <StatCard
          title="Global Datasets"
          value={stats.total_datasets}
          icon={Activity}
          color="bg-purple-500"
        />
        <StatCard
          title="Active Models"
          value={stats.total_models}
          icon={Shield}
          color="bg-orange-500"
        />
      </div>
    </div>
  );
};

export default Dashboard;
