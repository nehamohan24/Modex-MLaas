const BACKEND_PORTS = [5001, 5000, 5002];
let cachedURL = null;
const getBaseHost = () => {
  const hostname = window.location.hostname; 
  if (hostname.includes("devtunnels.ms")) {  
    return hostname.replace("3000", "5001");
  }
  return "localhost";
};

const detectBackendPort = async () => {
  const baseHost = getBaseHost();
  const isDevTunnel = baseHost.includes("devtunnels.ms");

  for (const port of BACKEND_PORTS) {
    // For devtunnel, don't include port in the URL
    const url = isDevTunnel
      ? `https://${baseHost}/api/domains`
      : `http://${baseHost}:${port}/api/domains`;

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2000);

    try {
      const response = await fetch(url, {
        method: "GET",
        mode: "cors",
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      if (response.ok) {
        console.log(`✅ Backend found at ${url}`);
        return { host: baseHost, port, isDevTunnel };
      }
    } catch (error) {
      clearTimeout(timeoutId);
      continue;
    }
  }

  console.warn("⚠️ No backend found on any port");
  return { host: baseHost, port: 5000, isDevTunnel };
};

export const getBackendURL = async () => {
  if (cachedURL) return cachedURL;

  const { host, port, isDevTunnel } = await detectBackendPort();
  cachedURL = isDevTunnel
    ? `https://${host}/api`
    : `http://${host}:${port}/api`;

  return cachedURL;
};

export default getBackendURL;
