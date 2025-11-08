import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { getBackendURL } from '../utils/api';
import { 
  Heart, 
  DollarSign, 
  Cpu, 
  Shield,
  CheckCircle,
  ArrowRight
} from 'lucide-react';

const DomainSelection = () => {
  const [domains, setDomains] = useState({});
  const [selectedDomain, setSelectedDomain] = useState('');
  const [selectedSubdomain, setSelectedSubdomain] = useState('');
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    fetchDomains();
  }, []);

  const fetchDomains = async () => {
    try {
      const backendURL = await getBackendURL();
      const response = await fetch(`${backendURL}/domains`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      console.log('Fetched domains:', data);
      setDomains(data);
    } catch (error) {
      console.error('Failed to fetch domains:', error);
      // Set empty domains object to prevent crashes
      setDomains({});
    } finally {
      setLoading(false);
    }
  };

  const domainIcons = {
    healthcare: Heart,
    finance: DollarSign,
    technology: Cpu
  };

  const handleDomainSelect = (domainKey) => {
    console.log('Domain selected:', domainKey);
    setSelectedDomain(domainKey);
    setSelectedSubdomain('');
  };

  const handleSubdomainSelect = (subdomain) => {
    console.log('Subdomain selected:', subdomain);
    setSelectedSubdomain(subdomain);
  };

  const handleContinue = () => {
    if (selectedDomain && selectedSubdomain) {
      // Store selection in localStorage
      localStorage.setItem('selectedDomain', selectedDomain);
      localStorage.setItem('selectedSubdomain', selectedSubdomain);
      navigate('/upload');
    }
  };

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
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          Choose Your Domain
        </h1>
        <p className="text-xl text-gray-600 max-w-3xl mx-auto">
          Select the domain and subdomain that best matches your data and research interests. 
          This will help us connect you with relevant collaborators and training opportunities.
        </p>
      </div>

      {/* Domain Selection */}
      <div className="mb-12">
        <h2 className="text-2xl font-semibold text-gray-900 mb-6">Select Domain</h2>
        {Object.keys(domains).length === 0 ? (
          <div className="text-center py-8">
            <p className="text-gray-500">No domains available. Please check your connection.</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {Object.entries(domains).map(([key, domain]) => {
            const Icon = domainIcons[key] || Shield;
            const isSelected = selectedDomain === key;
            
            return (
              <div
                key={key}
                onClick={() => handleDomainSelect(key)}
                className={`card cursor-pointer transition-all duration-200 hover:shadow-md ${
                  isSelected 
                    ? 'ring-2 ring-primary-500 bg-primary-50' 
                    : 'hover:shadow-md'
                }`}
              >
                <div className="flex items-center mb-4">
                  <div className={`p-3 rounded-lg ${
                    isSelected ? 'bg-primary-500' : 'bg-gray-100'
                  }`}>
                    <Icon className={`h-6 w-6 ${
                      isSelected ? 'text-white' : 'text-gray-600'
                    }`} />
                  </div>
                  <div className="ml-4">
                    <h3 className={`text-lg font-semibold ${
                      isSelected ? 'text-primary-700' : 'text-gray-900'
                    }`}>
                      {domain.name}
                    </h3>
                  </div>
                  {isSelected && (
                    <CheckCircle className="h-5 w-5 text-primary-500 ml-auto" />
                  )}
                </div>
                <p className="text-sm text-gray-600">
                  {Object.keys(domain.subdomains || {}).length} subdomains available
                </p>
              </div>
            );
          })}
          </div>
        )}
      </div>

      {/* Subdomain Selection */}
      {selectedDomain && (
        <div className="mb-12">
          <h2 className="text-2xl font-semibold text-gray-900 mb-6">
            Select Subdomain
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {domains[selectedDomain]?.subdomains && Object.entries(domains[selectedDomain].subdomains).map(([key, subdomain]) => {
              const isSelected = selectedSubdomain === key;
              
              return (
                <div
                  key={key}
                  onClick={() => handleSubdomainSelect(key)}
                  className={`p-4 rounded-lg border-2 cursor-pointer transition-all duration-200 ${
                    isSelected
                      ? 'border-primary-500 bg-primary-50 text-primary-700'
                      : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-medium capitalize">
                      {subdomain.name || key.replace('_', ' ')}
                    </span>
                    {isSelected && (
                      <CheckCircle className="h-5 w-5 text-primary-500" />
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Continue Button */}
      {selectedDomain && selectedSubdomain && (
        <div className="text-center">
          <button
            onClick={handleContinue}
            className="btn-primary text-lg px-8 py-3 inline-flex items-center"
          >
            Continue to Data Upload
            <ArrowRight className="ml-2 h-5 w-5" />
          </button>
        </div>
      )}

      {/* Domain Information */}
      {selectedDomain && (
        <div className="mt-12 card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            About {domains[selectedDomain]?.name}
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium text-gray-900 mb-2">Available Subdomains:</h4>
              <ul className="space-y-1">
                {domains[selectedDomain]?.subdomains && Object.entries(domains[selectedDomain].subdomains).map(([key, subdomain]) => (
                  <li key={key} className="text-sm text-gray-600">
                    • {subdomain.name || key.replace('_', ' ').toUpperCase()}
                  </li>
                ))}
              </ul>
              
              {/* Keywords for selected subdomain */}
              {selectedSubdomain && domains[selectedDomain]?.subdomains[selectedSubdomain]?.keywords && (
                <div className="mt-4">
                  <h4 className="font-medium text-gray-900 mb-2">
                    Expected Data Keywords for {domains[selectedDomain].subdomains[selectedSubdomain].name}:
                  </h4>
                  <div className="flex flex-wrap gap-1">
                    {domains[selectedDomain].subdomains[selectedSubdomain].keywords.slice(0, 8).map((keyword, index) => (
                      <span
                        key={index}
                        className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-blue-100 text-blue-800"
                      >
                        {keyword}
                      </span>
                    ))}
                    {domains[selectedDomain].subdomains[selectedSubdomain].keywords.length > 8 && (
                      <span className="text-xs text-gray-500">
                        +{domains[selectedDomain].subdomains[selectedSubdomain].keywords.length - 8} more
                      </span>
                    )}
                  </div>
                </div>
              )}
            </div>
            <div>
              <h4 className="font-medium text-gray-900 mb-2">Collaboration Benefits:</h4>
              <ul className="space-y-1 text-sm text-gray-600">
                <li>• Access to encrypted datasets from other researchers</li>
                <li>• Collaborative model training with privacy preservation</li>
                <li>• Federated learning across multiple institutions</li>
                <li>• Advanced encryption using Concrete-ML</li>
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DomainSelection;
