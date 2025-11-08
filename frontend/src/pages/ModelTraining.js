import React, { useState, useEffect } from 'react';
import { getBackendURL } from '../utils/api';
import { 
  Brain, 
  TrendingUp, 
  Shield, 
  Users,
  CheckCircle,
  AlertCircle,
  Zap
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import toast from 'react-hot-toast';

const ModelTraining = () => {
  const [selectedDomain, setSelectedDomain] = useState('');
  const [selectedSubdomain, setSelectedSubdomain] = useState('');
  const [modelType, setModelType] = useState('linear_regression');
  const [trainingStatus, setTrainingStatus] = useState('idle'); 
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [modelResults, setModelResults] = useState(null);
  const [domains, setDomains] = useState({});
  const [userDomainCount, setUserDomainCount] = useState(0);
  const [userSubdomainCount, setUserSubdomainCount] = useState(0);
  const [globalDomainCount, setGlobalDomainCount] = useState(0);
  const [globalSubdomainCount, setGlobalSubdomainCount] = useState(0);



  useEffect(() => {
    fetchDomains();
  }, []);
  useEffect(() => {
  if (!selectedDomain) return;
  fetchDatasetCounts(selectedDomain, selectedSubdomain);
}, [selectedDomain, selectedSubdomain]);

  
  useEffect(() => {
  if (selectedDomain) {
    fetchDatasetCounts(selectedDomain, selectedSubdomain);
  } else {
    setUserDomainCount(0);
    setUserSubdomainCount(0);
    setGlobalDomainCount(0);
    setGlobalSubdomainCount(0);
  }
}, [selectedDomain, selectedSubdomain]);

  const fetchDomains = async () => {
    try {
      const backendURL = await getBackendURL();
      const response = await fetch(`${backendURL}/domains`);
      const data = await response.json();
      setDomains(data);
    } catch (error) {
      console.error('Failed to fetch domains:', error);
    }
  };

  const fetchDatasetCounts = async (domain, subdomain = '') => {
  try {
    const token = localStorage.getItem('token');
    const backendURL = await getBackendURL();
    const url = new URL(`${backendURL}/datasets/counts`);
    url.searchParams.append('domain', domain);
    if (subdomain) url.searchParams.append('subdomain', subdomain);

    const response = await fetch(url.toString(), {
      headers: {
        'Authorization': `Bearer ${token}`,
      },
    });

    const data = await response.json();
    console.log('Fetched dataset counts:', data);

    // Map backend keys to frontend state
    setUserDomainCount(data.user_count ?? 0);
    setUserSubdomainCount(data.user_subdomain ?? data.user_count ?? 0);
    setGlobalDomainCount(data.global_count ?? 0);
    setGlobalSubdomainCount(data.global_subdomain ?? data.global_count ?? 0);
  } catch (error) {
    console.error('Failed to fetch dataset counts:', error);
    setUserDomainCount(0);
    setUserSubdomainCount(0);
    setGlobalDomainCount(0);
    setGlobalSubdomainCount(0);
  }
};




  const startTraining = async () => {
    if (!selectedDomain || !selectedSubdomain) {
      toast.error('Please select domain and subdomain');
      return;
    }

    setTrainingStatus('training');
    setTrainingProgress(0);

    // Simulate training progress
    const progressInterval = setInterval(() => {
      setTrainingProgress(prev => {
        if (prev >= 100) {
          clearInterval(progressInterval);
          return 100;
        }
        return prev + Math.random() * 10;
      });
    }, 500);

    try {
      const token = localStorage.getItem('token');
      const backendURL = await getBackendURL();
      const response = await fetch(`${backendURL}/train-model`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          domain: selectedDomain,
          subdomain: selectedSubdomain,
          model_type: modelType
        })
      });

      const result = await response.json();

      if (response.ok) {
        setModelResults(result);
        setTrainingStatus('completed');
        toast.success('Model training completed successfully!');
      } else {
        throw new Error(result.error || 'Training failed');
      }
    } catch (error) {
      setTrainingStatus('error');
      toast.error('Training failed: ' + error.message);
    }
  };

  const modelTypes = [
    {
      id: 'linear_regression',
      name: 'Linear Regression',
      description: 'Fast and interpretable model for continuous predictions',
      accuracy: '85-95%',
      icon: TrendingUp
    },
    {
      id: 'concrete_ml',
      name: 'Concrete-ML (Encrypted)',
      description: 'Homomorphically encrypted training for maximum privacy',
      accuracy: '80-90%',
      icon: Shield
    }
  ];

  const trainingSteps = [
    { step: 1, name: 'Data Collection', description: 'Gathering encrypted datasets from contributors' },
    { step: 2, name: 'Data Preprocessing', description: 'Cleaning and preparing data for training' },
    { step: 3, name: 'Model Training', description: 'Training the selected model algorithm' },
    { step: 4, name: 'Validation', description: 'Testing model performance and accuracy' },
    { step: 5, name: 'Encryption', description: 'Securing the trained model' }
  ];

  const performanceData = [
    { epoch: 1, accuracy: 75, loss: 0.8 },
    { epoch: 2, accuracy: 82, loss: 0.6 },
    { epoch: 3, accuracy: 87, loss: 0.4 },
    { epoch: 4, accuracy: 91, loss: 0.3 },
    { epoch: 5, accuracy: 94, loss: 0.2 },
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">
          Collaborative Model Training
        </h1>
        <p className="text-gray-600">
          Train ML models using encrypted data from multiple contributors while maintaining privacy.
        </p>
      </div>

      {/* Configuration */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        {/* Domain Selection */}
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Training Configuration</h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Domain</label>
              <select
                value={selectedDomain}
                onChange={(e) => setSelectedDomain(e.target.value)}
                className="input-field"
              >
                <option value="">Select a domain</option>
                {Object.entries(domains).map(([key, domain]) => (
                  <option key={key} value={key}>{domain.name}</option>
                ))}
              </select>
            </div>
            
            {selectedDomain && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Subdomain</label>
                <select
                  value={selectedSubdomain}
                  onChange={(e) => setSelectedSubdomain(e.target.value)}
                  className="input-field"
                >
                  <option value="">Select a subdomain</option>
                  {domains[selectedDomain]?.subdomains && Object.entries(domains[selectedDomain].subdomains).map(([key, subdomain]) => (
                    <option key={key} value={key}>
                      {subdomain.name || key.replace('_', ' ').toUpperCase()}
                    </option>
                  ))}
                </select>
              </div>
            )}

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Model Type</label>
              <div className="space-y-2">
                {modelTypes.map((type) => {
                  const Icon = type.icon;
                  return (
                    <label
                      key={type.id}
                      className={`flex items-center p-3 border rounded-lg cursor-pointer transition-colors duration-200 ${
                        modelType === type.id
                          ? 'border-primary-500 bg-primary-50'
                          : 'border-gray-200 hover:border-gray-300'
                      }`}
                    >
                      <input
                        type="radio"
                        name="modelType"
                        value={type.id}
                        checked={modelType === type.id}
                        onChange={(e) => setModelType(e.target.value)}
                        className="sr-only"
                      />
                      <Icon className="h-5 w-5 text-primary-600 mr-3" />
                      <div className="flex-1">
                        <div className="flex items-center justify-between">
                          <span className="font-medium text-gray-900">{type.name}</span>
                          <span className="text-sm text-green-600">{type.accuracy}</span>
                        </div>
                        <p className="text-sm text-gray-600">{type.description}</p>
                      </div>
                    </label>
                  );
                })}
              </div>
            </div>
          </div>
        </div>

        {/* Training Status */}
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Training Status</h2>
          
          {trainingStatus === 'idle' && (
            <div className="text-center py-8">
              <Brain className="mx-auto h-12 w-12 text-gray-300 mb-4" />
              <p className="text-gray-500">Ready to start training</p>
              <p className="text-sm text-gray-400 mt-2">
                  Your datasets: {userSubdomainCount} 
                </p>
                <p className="text-sm text-gray-400">
                  Global datasets: {globalSubdomainCount} 
                </p>
            </div>
          )}

          {trainingStatus === 'training' && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-gray-700">Training Progress</span>
                <span className="text-sm text-gray-500">{Math.round(trainingProgress)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-primary-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${trainingProgress}%` }}
                ></div>
              </div>
              
              <div className="space-y-2">
                {trainingSteps.map((step) => (
                  <div key={step.step} className="flex items-center space-x-3">
                    <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs ${
                      trainingProgress >= (step.step * 20)
                        ? 'bg-green-500 text-white'
                        : 'bg-gray-200 text-gray-500'
                    }`}>
                      {trainingProgress >= (step.step * 20) ? (
                        <CheckCircle className="h-4 w-4" />
                      ) : (
                        step.step
                      )}
                    </div>
                    <div>
                      <p className="text-sm font-medium text-gray-900">{step.name}</p>
                      <p className="text-xs text-gray-500">{step.description}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {trainingStatus === 'completed' && modelResults && (
            <div className="space-y-4">
              <div className="flex items-center space-x-2 text-green-600">
                <CheckCircle className="h-5 w-5" />
                <span className="font-medium">Training Completed!</span>
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div className="p-3 bg-green-50 rounded-lg">
                  <p className="text-sm text-gray-600">Accuracy</p>
                  <p className="text-2xl font-bold text-green-600">
                    {(modelResults.accuracy * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="p-3 bg-blue-50 rounded-lg">
                  <p className="text-sm text-gray-600">R² Score</p>
                  <p className="text-2xl font-bold text-blue-600">
                    {modelResults.r2_score.toFixed(3)}
                  </p>
                </div>
              </div>
              
              <div className="text-sm text-gray-600">
                <p>Contributors: {modelResults.contributors_count}</p>
                <p>Training Data: {modelResults.training_data_count} records</p>
              </div>
            </div>
          )}

          {trainingStatus === 'error' && (
            <div className="flex items-center space-x-2 text-red-600">
              <AlertCircle className="h-5 w-5" />
              <span className="font-medium">Training Failed</span>
            </div>
          )}

          {/* Start Training Button */}
          {trainingStatus === 'idle' && (
            <button
              onClick={startTraining}
              disabled={!selectedDomain || !selectedSubdomain}
              className="w-full btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {/* <Play className="h-4 w-10 mr-4" /> */}
              Start Training
            </button>
          )}
        </div>
      </div>

      {/* Performance Chart */}
      {trainingStatus === 'training' && (
        <div className="card mb-8">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Training Performance</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={performanceData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="epoch" />
              <YAxis />
              <Tooltip />
              <Line 
                type="monotone" 
                dataKey="accuracy" 
                stroke="#3B82F6" 
                strokeWidth={2}
                name="Accuracy %"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Training Information */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="card">
          <div className="flex items-center mb-4">
            <Shield className="h-6 w-6 text-primary-600 mr-2" />
            <h3 className="font-semibold text-gray-900">Privacy Protection</h3>
          </div>
          <p className="text-sm text-gray-600">
            Your data remains encrypted throughout the training process using advanced homomorphic encryption techniques.
          </p>
        </div>

        <div className="card">
          <div className="flex items-center mb-4">
            <Users className="h-6 w-6 text-green-600 mr-2" />
            <h3 className="font-semibold text-gray-900">Collaborative Learning</h3>
          </div>
          <p className="text-sm text-gray-600">
            Models are trained using federated learning, combining insights from multiple contributors without sharing raw data.
          </p>
        </div>

        <div className="card">
          <div className="flex items-center mb-4">
            <Zap className="h-6 w-6 text-yellow-600 mr-2" />
            <h3 className="font-semibold text-gray-900">High Accuracy</h3>
          </div>
          <p className="text-sm text-gray-600">
            Our collaborative approach typically achieves 90%+ accuracy by leveraging diverse datasets from multiple sources.
          </p>
        </div>
      </div>
    </div>
  );
};

export default ModelTraining;
