// Webpack configuration override to fix allowedHosts warning
const path = require('path');

module.exports = {
  devServer: {
    allowedHosts: ['localhost', '127.0.0.1'],
    host: 'localhost',
    port: 3000,
    hot: true,
    open: true,
    historyApiFallback: true,
    client: {
      webSocketURL: 'ws://localhost:3000/ws',
    },
  },
};
