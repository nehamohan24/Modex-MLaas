import React, { useState, useEffect } from 'react';
import { getBackendURL } from '../utils/api';
import { 
  Brain, 
  Eye, 
  Download, 
  Share2, 
  TrendingUp, 
  Users,
  Clock,
  Shield,
  Zap,
  BarChart3
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import toast from 'react-hot-toast';

const ModelViewer = () => {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);
  const [loading, setLoading] = useState(true);
  const [predictionInput, setPredictionInput] = useState('');
  const [predictionResult, setPredictionResult] = useState(null);
  const [selectedDomain, setSelectedDomain] = useState('');
  const [selectedSubdomain, setSelectedSubdomain] = useState('');
  const [modelAttributes, setModelAttributes] = useState([]);
  const [inputValues, setInputValues] = useState({});
  const [loadingAttributes, setLoadingAttributes] = useState(false);
  const [targetLabel, setTargetLabel] = useState('target');
  const [targetDescription, setTargetDescription] = useState('predicted outcome');

  // Determine attribute input type based on common attribute names
  const isTextAttribute = (attrName) => {
    const name = String(attrName || '').toLowerCase();
    const textLikeAttributes = new Set([
      'gender',
      'smoking',
      'alcohol',
      'family_history',
      'treatment_type',
      'genetic_markers',
      'medication',
      'symptoms_duration',
      'neurological_exam',
      'imaging_results',
      'education_level',
      'employment_status',
      'payment_history',
      'coverage_type',
      'occupation',
      'location_risk',
      'sector',
      'device_type',
      'region',
      'ip_address',
      'geolocation',
    ]);
    // Any attribute containing common categorical tokens
    const categoricalHints = ['type', 'status', 'history', 'grade', 'stage'];
    if (textLikeAttributes.has(name)) return true;
    if (categoricalHints.some((hint) => name.includes(hint))) return true;
    return false;
  };

  useEffect(() => {
    fetchModels();
  }, [selectedDomain, selectedSubdomain]);

  const fetchModels = async () => {
    try {
      if (!selectedDomain || !selectedSubdomain) {
        setLoading(false);
        return;
      }

      const token = localStorage.getItem('token');
      const backendURL = await getBackendURL();
      const response = await fetch(
        `${backendURL}/models/${selectedDomain}/${selectedSubdomain}`,
        {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        }
      );
      
      if (response.ok) {
        const data = await response.json();
        setModels(data);
      }
    } catch (error) {
      console.error('Failed to fetch models:', error);
      toast.error('Failed to fetch models');
    } finally {
      setLoading(false);
    }
  };

  const fetchModelAttributes = async (modelId) => {
    try {
      setLoadingAttributes(true);
      const token = localStorage.getItem('token');
      const backendURL = await getBackendURL();
      const response = await fetch(`${backendURL}/model/${modelId}/attributes`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.ok) {
        const data = await response.json();
        setModelAttributes(data.required_attributes);
        setTargetLabel(data.target_label || 'target');
        setTargetDescription(data.target_description || 'predicted outcome');
        // Initialize input values
        const initialValues = {};
        data.required_attributes.forEach(attr => {
          initialValues[attr] = '';
        });
        setInputValues(initialValues);
      } else {
        throw new Error('Failed to fetch model attributes');
      }
    } catch (error) {
      console.error('Failed to fetch model attributes:', error);
      toast.error('Failed to fetch model attributes');
    } finally {
      setLoadingAttributes(false);
    }
  };

  const makePrediction = async (modelId, inputData) => {
    try {
      const token = localStorage.getItem('token');
      const backendURL = await getBackendURL();
      const response = await fetch(`${backendURL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          model_id: modelId,
          input_data: inputData
        })
      });

      if (response.ok) {
        const result = await response.json();
        setPredictionResult(result);
        if (result.target_label) {
          setTargetLabel(result.target_label);
        }
        if (result.target_description) {
          setTargetDescription(result.target_description);
        }
        toast.success('Prediction completed!');
      } else {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Prediction failed');
      }
    } catch (error) {
      toast.error('Prediction failed: ' + error.message);
    }
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getAccuracyColor = (accuracy) => {
    if (accuracy >= 0.9) return 'text-green-600';
    if (accuracy >= 0.8) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getAccuracyBgColor = (accuracy) => {
    if (accuracy >= 0.9) return 'bg-green-100';
    if (accuracy >= 0.8) return 'bg-yellow-100';
    return 'bg-red-100';
  };

  const performanceData = [
    { metric: 'Accuracy', value: 94, color: '#10B981' },
    { metric: 'Precision', value: 92, color: '#3B82F6' },
    { metric: 'Recall', value: 96, color: '#F59E0B' },
    { metric: 'F1-Score', value: 94, color: '#8B5CF6' },
  ];

  const trainingHistory = [
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
          Model Explorer
        </h1>
        <p className="text-gray-600">
          Explore, analyze, and use trained models from collaborative ML training sessions.
        </p>
      </div>

      {/* Domain Filter */}
      <div className="card mb-8">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Filter Models</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Domain</label>
            <select
              value={selectedDomain}
              onChange={(e) => setSelectedDomain(e.target.value)}
              className="input-field"
            >
              <option value="">All Domains</option>
              <option value="healthcare">Healthcare</option>
              <option value="finance">Finance</option>
              <option value="technology">Technology</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Subdomain</label>
            <select
              value={selectedSubdomain}
              onChange={(e) => setSelectedSubdomain(e.target.value)}
              className="input-field"
              disabled={!selectedDomain}
            >
              <option value="">All Subdomains</option>
              {selectedDomain === 'healthcare' && [
                <option key="cardiology" value="cardiology">Cardiology</option>,
                <option key="oncology" value="oncology">Oncology</option>,
                <option key="neurology" value="neurology">Neurology</option>
              ]}
              {selectedDomain === 'finance' && [
                <option key="banking" value="banking">Banking</option>,
                <option key="insurance" value="insurance">Insurance</option>,
                <option key="trading" value="trading">Trading</option>
              ]}
              {selectedDomain === 'technology' && [
                <option key="ai_ml" value="ai_ml">AI/ML</option>,
                <option key="cybersecurity" value="cybersecurity">Cybersecurity</option>,
                <option key="cloud_computing" value="cloud_computing">Cloud Computing</option>
              ]}
            </select>
          </div>
        </div>
      </div>

      {/* Models Grid */}
      {loading ? (
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
        </div>
      ) : models.length === 0 ? (
        <div className="text-center py-12">
          <Brain className="mx-auto h-12 w-12 text-gray-300 mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No Models Found</h3>
          <p className="text-gray-600">
            {selectedDomain && selectedSubdomain 
              ? `No models available for ${selectedDomain} - ${selectedSubdomain}`
              : 'Please select a domain and subdomain to view models'
            }
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {models.map((model) => (
            <div key={model._id} className="card hover:shadow-md transition-shadow duration-200">
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center">
                  <div className="p-2 rounded-lg bg-primary-100">
                    <Brain className="h-5 w-5 text-primary-600" />
                  </div>
                  <div className="ml-3">
                    <h3 className="font-semibold text-gray-900">
                      {model.model_type.replace('_', ' ').toUpperCase()}
                    </h3>
                    <p className="text-sm text-gray-600">
                      {model.domain} - {model.subdomain}
                    </p>
                  </div>
                </div>
                <div className={`px-2 py-1 rounded-full text-xs font-medium ${getAccuracyBgColor(model.accuracy)} ${getAccuracyColor(model.accuracy)}`}>
                  {(model.accuracy * 100).toFixed(1)}%
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4 mb-4">
                <div className="text-center p-3 bg-gray-50 rounded-lg">
                  <p className="text-sm text-gray-600">R² Score</p>
                  <p className="text-lg font-semibold text-gray-900">{model.r2_score.toFixed(3)}</p>
                </div>
                <div className="text-center p-3 bg-gray-50 rounded-lg">
                  <p className="text-sm text-gray-600">Contributors</p>
                  <p className="text-lg font-semibold text-gray-900">{model.contributors.length}</p>
                </div>
              </div>

              <div className="flex items-center justify-between text-sm text-gray-500 mb-4">
                <div className="flex items-center">
                  <Clock className="h-4 w-4 mr-1" />
                  {formatDate(model.created_at)}
                </div>
                <div className="flex items-center">
                  <Users className="h-4 w-4 mr-1" />
                  {model.training_data_count} records
                </div>
              </div>

              <div className="flex space-x-2">
                <button
                  onClick={() => setSelectedModel(model)}
                  className="flex-1 btn-secondary text-sm"
                >
                  <Eye className="h-4 w-4 mr-1" />
                  View Details
                </button>
                <button
                  onClick={() => {
                    setSelectedModel(model);
                    fetchModelAttributes(model._id);
                  }}
                  className="flex-1 btn-primary text-sm"
                >
                  <Zap className="h-4 w-4 mr-1" />
                  Test Model
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Model Details Modal */}
      {selectedModel && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-4xl max-h-[90vh] overflow-hidden">
            <div className="flex items-center justify-between p-6 border-b">
              <h2 className="text-xl font-semibold text-gray-900">Model Details</h2>
              <button
                onClick={() => setSelectedModel(null)}
                className="text-gray-400 hover:text-gray-600"
              >
                ×
              </button>
            </div>
            
            <div className="p-6 overflow-auto max-h-[70vh]">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Model Info */}
                <div className="space-y-4">
                  <div>
                    <h3 className="font-semibold text-gray-900 mb-2">Model Information</h3>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-600">Type:</span>
                        <span className="font-medium">{selectedModel.model_type}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Domain:</span>
                        <span className="font-medium">{selectedModel.domain}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Subdomain:</span>
                        <span className="font-medium">{selectedModel.subdomain}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Created:</span>
                        <span className="font-medium">{formatDate(selectedModel.created_at)}</span>
                      </div>
                    </div>
                  </div>

                  <div>
                    <h3 className="font-semibold text-gray-900 mb-2">Performance Metrics</h3>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-gray-600">Accuracy:</span>
                        <span className={`font-medium ${getAccuracyColor(selectedModel.accuracy)}`}>
                          {(selectedModel.accuracy * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">R² Score:</span>
                        <span className="font-medium">{selectedModel.r2_score.toFixed(3)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">MSE:</span>
                        <span className="font-medium">{selectedModel.mse.toFixed(4)}</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Performance Chart */}
                <div>
                  <h3 className="font-semibold text-gray-900 mb-4">Training Performance</h3>
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={trainingHistory}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="epoch" />
                      <YAxis />
                      <Tooltip />
                      <Line 
                        type="monotone" 
                        dataKey="accuracy" 
                        stroke="#3B82F6" 
                        strokeWidth={2}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Prediction Interface */}
              <div className="mt-6 pt-6 border-t">
                <h3 className="font-semibold text-gray-900 mb-1">Test Model</h3>
                <p className="text-sm text-gray-600 mb-4">Predicting: <span className="font-medium text-gray-800">{targetDescription}</span></p>
                
                {loadingAttributes ? (
                  <div className="flex items-center justify-center py-8">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
                    <span className="ml-2 text-gray-600">Loading model attributes...</span>
                  </div>
                ) : modelAttributes.length > 0 ? (
                  <div className="space-y-4">
                    <div className="mb-4">
                      <h4 className="text-sm font-medium text-gray-700 mb-2">
                        Required Attributes ({modelAttributes.length}):
                      </h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {modelAttributes.map((attribute, index) => (
                          <div key={attribute}>
                            <label className="block text-sm font-medium text-gray-700 mb-1">
                              {attribute}
                            </label>
                            <input
                              type={isTextAttribute(attribute) ? 'text' : 'number'}
                              step={isTextAttribute(attribute) ? undefined : 'any'}
                              placeholder={`Enter ${attribute}`}
                              value={inputValues[attribute] || ''}
                              onChange={(e) => setInputValues(prev => ({
                                ...prev,
                                [attribute]: e.target.value
                              }))}
                              className="input-field"
                            />
                          </div>
                        ))}
                      </div>
                    </div>
                    
                    <button
                      onClick={() => {
                        const inputArray = modelAttributes.map(attr => {
                          const raw = inputValues[attr];
                          if (isTextAttribute(attr)) {
                            return String(raw ?? '');
                          }
                          const num = parseFloat(raw);
                          return Number.isFinite(num) ? num : 0;
                        });
                        makePrediction(selectedModel._id, inputArray);
                      }}
                      disabled={modelAttributes.some(attr => {
                        const v = inputValues[attr];
                        if (v === undefined || v === null || v === '') return true;
                        if (!isTextAttribute(attr)) {
                          const n = parseFloat(v);
                          if (!Number.isFinite(n)) return true;
                        }
                        return false;
                      })}
                      className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      <Zap className="h-4 w-4 mr-2" />
                      Make Prediction
                    </button>
                  </div>
                ) : (
                  <div className="text-center py-8 text-gray-500">
                    <p>No attributes available for this model.</p>
                  </div>
                )}
                
                {predictionResult && (
                  <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-lg">
                    <h4 className="font-medium text-green-800 mb-3">Prediction Result</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <p className="text-sm text-green-600">Predicted {targetDescription}:</p>
                        <p className="text-lg font-semibold text-green-800">
                          {predictionResult.prediction}
                        </p>
                      </div>
                      <div>
                        <p className="text-sm text-green-600">Confidence:</p>
                        <p className="text-lg font-semibold text-green-800">
                          {(predictionResult.confidence * 100).toFixed(1)}%
                        </p>
                      </div>
                    </div>
                    <div className="mt-3 pt-3 border-t border-green-200">
                      <p className="text-sm text-green-600">
                        Model Accuracy: {(predictionResult.model_accuracy * 100).toFixed(1)}%
                      </p>
                      <p className="text-sm text-green-600">
                        Model Type: {predictionResult.model_type}
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="card text-center">
          <div className="p-3 rounded-lg bg-blue-100 mx-auto w-fit mb-3">
            <Brain className="h-6 w-6 text-blue-600" />
          </div>
          <h3 className="font-semibold text-gray-900">Total Models</h3>
          <p className="text-2xl font-bold text-blue-600">{models.length}</p>
        </div>

        <div className="card text-center">
          <div className="p-3 rounded-lg bg-green-100 mx-auto w-fit mb-3">
            <TrendingUp className="h-6 w-6 text-green-600" />
          </div>
          <h3 className="font-semibold text-gray-900">Avg Accuracy</h3>
          <p className="text-2xl font-bold text-green-600">
            {models.length > 0 
              ? ((models.reduce((sum, m) => sum + m.accuracy, 0) / models.length) * 100).toFixed(1)
              : 0
            }%
          </p>
        </div>

        <div className="card text-center">
          <div className="p-3 rounded-lg bg-purple-100 mx-auto w-fit mb-3">
            <Users className="h-6 w-6 text-purple-600" />
          </div>
          <h3 className="font-semibold text-gray-900">Contributors</h3>
          <p className="text-2xl font-bold text-purple-600">
            {models.length > 0 
              ? Math.max(...models.map(m => m.contributors.length))
              : 0
            }
          </p>
        </div>

        <div className="card text-center">
          <div className="p-3 rounded-lg bg-orange-100 mx-auto w-fit mb-3">
            <Shield className="h-6 w-6 text-orange-600" />
          </div>
          <h3 className="font-semibold text-gray-900">Encrypted</h3>
          <p className="text-2xl font-bold text-orange-600">100%</p>
        </div>
      </div>
    </div>
  );
};

export default ModelViewer;
