import React, { useState, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { getBackendURL } from '../utils/api';
import { 
  Upload, 
  File, 
  Shield, 
  CheckCircle, 
  AlertCircle,
  Database,
  Lock,
  Eye,
  EyeOff
} from 'lucide-react';
import toast from 'react-hot-toast';

const DataUpload = () => {
  const [selectedDomain, setSelectedDomain] = useState('');
  const [selectedSubdomain, setSelectedSubdomain] = useState('');
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [dataPreview, setDataPreview] = useState(null);
  const [showPreview, setShowPreview] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [domains, setDomains] = useState({});

  useEffect(() => {
   
    const domain = localStorage.getItem('selectedDomain');
    const subdomain = localStorage.getItem('selectedSubdomain');
    if (domain && subdomain) {
      setSelectedDomain(domain);
      setSelectedSubdomain(subdomain);
    }
    fetchDomains();
  }, []);

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

  const onDrop = (acceptedFiles) => {
    const newFiles = acceptedFiles.map(file => ({
      file,
      id: Math.random().toString(36).substr(2, 9),
      name: file.name,
      size: file.size,
      type: file.type,
      status: 'pending'
    }));
    setUploadedFiles(prev => [...prev, ...newFiles]);
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/json': ['.json'],
      'text/plain': ['.txt'],
      'application/vnd.ms-excel': ['.xls'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx']
    },
    multiple: true
  });

  const removeFile = (fileId) => {
    setUploadedFiles(prev => prev.filter(f => f.id !== fileId));
  };

  const previewFile = async (file) => {
    try {
      const text = await file.file.text();
      let parsedData;
      
      if (file.type === 'application/json') {
        parsedData = JSON.parse(text);
      } else if (file.type === 'text/csv') {
        const lines = text.split('\n');
        const headers = lines[0].split(',');
        parsedData = lines.slice(1, 6).map(line => {
          const values = line.split(',');
          const obj = {};
          headers.forEach((header, index) => {
            obj[header.trim()] = values[index]?.trim();
          });
          return obj;
        });
      } else {
        parsedData = text.split('\n').slice(0, 10);
      }
      
      setDataPreview(parsedData);
      setShowPreview(true);
    } catch (error) {
      toast.error('Failed to preview file');
    }
  };

  const uploadFiles = async () => {
    if (!selectedDomain || !selectedSubdomain) {
      toast.error('Please select domain and subdomain first');
      return;
    }

    if (uploadedFiles.length === 0) {
      toast.error('Please select files to upload');
      return;
    }

    setUploading(true);
    
    try {
      const token = localStorage.getItem('token');
      
      for (const fileInfo of uploadedFiles) {
        const formData = new FormData();
        formData.append('file', fileInfo.file);
        formData.append('domain', selectedDomain);
        formData.append('subdomain', selectedSubdomain);
        
        const backendURL = await getBackendURL();
        const response = await fetch(`${backendURL}/upload-dataset`, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`
          },
          body: formData
        });
        
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || 'Upload failed');
        }
        
        
        setUploadedFiles(prev => 
          prev.map(f => 
            f.id === fileInfo.id 
              ? { ...f, status: 'completed' }
              : f
          )
        );
      }
      
      toast.success('Files uploaded successfully!');
      setUploadedFiles([]);
      
    } catch (error) {
      toast.error('Upload failed: ' + error.message);
    } finally {
      setUploading(false);
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">
          Upload Your Data
        </h1>
        <p className="text-gray-600">
          Securely upload your encrypted datasets to contribute to collaborative ML training.
        </p>
        {selectedDomain && selectedSubdomain && (
          <div className="mt-4">
            <div className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-primary-100 text-primary-800 mb-4">
              <Database className="h-4 w-4 mr-2" />
              {domains[selectedDomain]?.subdomains[selectedSubdomain]?.name || selectedSubdomain.replace('_', ' ').toUpperCase()}
            </div>
            
            {/* Keywords Display */}
            {domains[selectedDomain]?.subdomains[selectedSubdomain]?.keywords && (
              <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg p-6 mb-6">
                <div className="flex items-center mb-3">
                  <div className="p-2 bg-blue-100 rounded-lg mr-3">
                    <Database className="h-5 w-5 text-blue-600" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-blue-900">
                      📋 Expected Data Keywords
                    </h3>
                    <p className="text-sm text-blue-700">
                      Your dataset should contain data related to these keywords for better validation
                    </p>
                  </div>
                </div>
                <div className="flex flex-wrap gap-2">
                  {domains[selectedDomain].subdomains[selectedSubdomain].keywords.map((keyword, index) => (
                    <span
                      key={index}
                      className="inline-flex items-center px-3 py-1.5 rounded-full text-sm font-medium bg-blue-100 text-blue-800 border border-blue-200 hover:bg-blue-200 transition-colors"
                    >
                      {keyword}
                    </span>
                  ))}
                </div>
                <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-md">
                  <p className="text-sm text-yellow-800">
                    <strong>💡 Tip:</strong> Include these keywords in your dataset's column names, file names, or content for better validation results.
                  </p>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Domain Selection */}
      {(!selectedDomain || !selectedSubdomain) && (
        <div className="card mb-8">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Select Domain & Subdomain</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
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
          </div>
          
          {/* Domain Keywords Preview */}
          {selectedDomain && !selectedSubdomain && domains[selectedDomain]?.subdomains && (
            <div className="mt-6 p-4 bg-gray-50 border border-gray-200 rounded-lg">
              <h3 className="text-sm font-medium text-gray-700 mb-3">
                📋 Available Subdomains & Keywords
              </h3>
              <div className="space-y-3">
                {Object.entries(domains[selectedDomain].subdomains).map(([key, subdomain]) => (
                  <div key={key} className="border border-gray-200 rounded-md p-3 bg-white">
                    <h4 className="font-medium text-gray-900 mb-2">
                      {subdomain.name || key.replace('_', ' ').toUpperCase()}
                    </h4>
                    {subdomain.keywords && (
                      <div className="flex flex-wrap gap-1">
                        {subdomain.keywords.slice(0, 6).map((keyword, index) => (
                          <span
                            key={index}
                            className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-blue-100 text-blue-800"
                          >
                            {keyword}
                          </span>
                        ))}
                        {subdomain.keywords.length > 6 && (
                          <span className="text-xs text-gray-500">
                            +{subdomain.keywords.length - 6} more
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Upload Area */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Dropzone */}
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Upload Files</h2>
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors duration-200 ${
              isDragActive
                ? 'border-primary-500 bg-primary-50'
                : 'border-gray-300 hover:border-gray-400'
            }`}
          >
            <input {...getInputProps()} />
            <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
            {isDragActive ? (
              <p className="text-lg text-primary-600">Drop the files here...</p>
            ) : (
              <div>
                <p className="text-lg text-gray-600 mb-2">
                  Drag & drop files here, or click to select
                </p>
                <p className="text-sm text-gray-500">
                  Supports CSV, JSON, TXT, XLS, XLSX files
                </p>
              </div>
            )}
          </div>

          {/* Keywords Reference */}
          {selectedDomain && selectedSubdomain && domains[selectedDomain]?.subdomains[selectedSubdomain]?.keywords && (
            <div className="mt-6 p-4 bg-amber-50 border border-amber-200 rounded-lg">
              <div className="flex items-start">
                <Database className="h-5 w-5 text-amber-600 mt-0.5 mr-3" />
                <div>
                  <h3 className="text-sm font-medium text-amber-800">Quick Keywords Reference</h3>
                  <p className="text-sm text-amber-700 mt-1 mb-2">
                    Make sure your dataset contains these keywords for better validation:
                  </p>
                  <div className="flex flex-wrap gap-1">
                    {domains[selectedDomain].subdomains[selectedSubdomain].keywords.slice(0, 8).map((keyword, index) => (
                      <span
                        key={index}
                        className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-amber-100 text-amber-800"
                      >
                        {keyword}
                      </span>
                    ))}
                    {domains[selectedDomain].subdomains[selectedSubdomain].keywords.length > 8 && (
                      <span className="text-xs text-amber-600">
                        +{domains[selectedDomain].subdomains[selectedSubdomain].keywords.length - 8} more
                      </span>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Security Notice */}
          <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-lg">
            <div className="flex items-start">
              <Shield className="h-5 w-5 text-green-600 mt-0.5 mr-3" />
              <div>
                <h3 className="text-sm font-medium text-green-800">Data Security</h3>
                <p className="text-sm text-green-700 mt-1">
                  Your data is encrypted using AES-256 encryption before storage and transmission. 
                  Only you and authorized collaborators can access your data.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* File List */}
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Uploaded Files</h2>
          {uploadedFiles.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              <File className="mx-auto h-12 w-12 text-gray-300 mb-4" />
              <p>No files uploaded yet</p>
            </div>
          ) : (
            <div className="space-y-3">
              {uploadedFiles.map((fileInfo) => (
                <div
                  key={fileInfo.id}
                  className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
                >
                  <div className="flex items-center space-x-3">
                    <File className="h-5 w-5 text-gray-400" />
                    <div>
                      <p className="text-sm font-medium text-gray-900">{fileInfo.name}</p>
                      <p className="text-xs text-gray-500">{formatFileSize(fileInfo.size)}</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => previewFile(fileInfo)}
                      className="p-1 text-gray-400 hover:text-gray-600"
                    >
                      <Eye className="h-4 w-4" />
                    </button>
                    <button
                      onClick={() => removeFile(fileInfo.id)}
                      className="p-1 text-red-400 hover:text-red-600"
                    >
                      ×
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}

          {uploadedFiles.length > 0 && (
            <div className="mt-6">
              <button
                onClick={uploadFiles}
                disabled={uploading}
                className="w-full btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {uploading ? (
                  <div className="flex items-center justify-center">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    Uploading...
                  </div>
                ) : (
                  'Upload Files'
                )}
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Data Preview Modal */}
      {showPreview && dataPreview && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-4xl max-h-[80vh] overflow-hidden">
            <div className="flex items-center justify-between p-4 border-b">
              <h3 className="text-lg font-semibold text-gray-900">Data Preview</h3>
              <button
                onClick={() => setShowPreview(false)}
                className="text-gray-400 hover:text-gray-600"
              >
                <EyeOff className="h-5 w-5" />
              </button>
            </div>
            <div className="p-4 overflow-auto max-h-96">
              <pre className="text-sm text-gray-700 whitespace-pre-wrap">
                {JSON.stringify(dataPreview, null, 2)}
              </pre>
            </div>
          </div>
        </div>
      )}

      {/* Upload Guidelines */}
      <div className="mt-8 card">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Upload Guidelines</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h3 className="font-medium text-gray-900 mb-2">Supported Formats</h3>
            <ul className="space-y-1 text-sm text-gray-600">
              <li>• CSV files (.csv)</li>
              <li>• JSON files (.json)</li>
              <li>• Excel files (.xls, .xlsx)</li>
              <li>• Text files (.txt)</li>
            </ul>
          </div>
          <div>
            <h3 className="font-medium text-gray-900 mb-2">Data Requirements</h3>
            <ul className="space-y-1 text-sm text-gray-600">
              <li>• Minimum 100 records recommended</li>
              <li>• Clear column headers</li>
              <li>• No personally identifiable information</li>
              <li>• Data should be relevant to selected domain</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DataUpload;
