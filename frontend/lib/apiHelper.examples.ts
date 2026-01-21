/**
 * Test file demonstrating apiHelper.ts usage
 * 
 * This file shows how to use the API helper utilities
 * to prevent common API mistakes in development.
 */

import {
  validateApiUrl,
  getProxyUrl,
  isProxyRoute,
  safeFetch,
  isApiValidationError,
} from './apiHelper';

// ========================================
// Example 1: Validate URLs before using
// ========================================

// ✅ CORRECT - using proxy route
try {
  const validUrl = validateApiUrl('/api/backend/bots/page');
  console.log('✅ Valid URL:', validUrl);
} catch (error) {
  console.error('❌ Invalid URL:', error);
}

// ❌ INCORRECT - hardcoded localhost (will throw in development)
try {
  const invalidUrl = validateApiUrl('http://localhost:8000/api/bots/page');
  console.log('Valid URL:', invalidUrl);
} catch (error) {
  if (isApiValidationError(error)) {
    console.error('❌ API Validation Error:', error.message);
  }
}

// ❌ INCORRECT - hardcoded IP address (will throw in development)
try {
  const invalidUrl = validateApiUrl('http://209.126.82.160:8000/api/bots/page');
  console.log('Valid URL:', invalidUrl);
} catch (error) {
  if (isApiValidationError(error)) {
    console.error('❌ API Validation Error:', error.message);
  }
}

// ❌ INCORRECT - double /api prefix (will throw in development)
try {
  const invalidUrl = validateApiUrl('/api/backend/api/bots/page');
  console.log('Valid URL:', invalidUrl);
} catch (error) {
  if (isApiValidationError(error)) {
    console.error('❌ API Validation Error:', error.message);
  }
}

// ========================================
// Example 2: Construct proxy URLs
// ========================================

// Get proxy URL for main backend
const botsUrl = getProxyUrl('/bots/page'); // Returns: /api/backend/bots/page
const statusUrl = getProxyUrl('system/status'); // Returns: /api/backend/system/status

// Get proxy URL for DT backend
const dtMetricsUrl = getProxyUrl('/learning/metrics', 'dt'); // Returns: /api/dt/learning/metrics

console.log('Main backend URLs:');
console.log('  Bots:', botsUrl);
console.log('  Status:', statusUrl);
console.log('DT backend URLs:');
console.log('  Metrics:', dtMetricsUrl);

// ========================================
// Example 3: Check if URL is proxy route
// ========================================

console.log('Is proxy route?');
console.log('  /api/backend/bots/page:', isProxyRoute('/api/backend/bots/page')); // true
console.log('  /api/dt/learning:', isProxyRoute('/api/dt/learning')); // true
console.log('  http://localhost:8000:', isProxyRoute('http://localhost:8000')); // false

// ========================================
// Example 4: Safe fetch with validation
// ========================================

async function fetchBotsData() {
  try {
    // Use safeFetch instead of fetch for automatic validation
    const response = await safeFetch('/api/backend/bots/page', {
      cache: 'no-store',
    });
    
    if (!response.ok) {
      throw new Error(`Failed to fetch: ${response.status}`);
    }
    
    const data = await response.json();
    console.log('Bots data:', data);
    return data;
  } catch (error) {
    if (isApiValidationError(error)) {
      console.error('API validation failed:', error.message);
    } else {
      console.error('Fetch failed:', error);
    }
    throw error;
  }
}

// ========================================
// Example 5: Using in React components
// ========================================

/*
// ❌ DON'T DO THIS:
export function BadComponent() {
  const [data, setData] = useState(null);
  
  useEffect(() => {
    // This will fail in development due to hardcoded URL
    fetch('http://localhost:8000/api/bots/page')
      .then(res => res.json())
      .then(setData);
  }, []);
  
  return <div>{JSON.stringify(data)}</div>;
}

// ✅ DO THIS INSTEAD:
import { safeFetch, getProxyUrl } from '@/lib/apiHelper';

export function GoodComponent() {
  const [data, setData] = useState(null);
  
  useEffect(() => {
    // Use proxy route - works with any backend
    safeFetch(getProxyUrl('/bots/page'), { 
      cache: 'no-store' 
    })
      .then(res => res.json())
      .then(setData)
      .catch(console.error);
  }, []);
  
  return <div>{JSON.stringify(data)}</div>;
}

// ✅ OR EVEN SIMPLER:
export function SimpleComponent() {
  const [data, setData] = useState(null);
  
  useEffect(() => {
    // Direct proxy route usage (recommended)
    fetch('/api/backend/bots/page', { cache: 'no-store' })
      .then(res => res.json())
      .then(setData)
      .catch(console.error);
  }, []);
  
  return <div>{JSON.stringify(data)}</div>;
}
*/

export {
  // Re-export for testing
  validateApiUrl,
  getProxyUrl,
  isProxyRoute,
  safeFetch,
  isApiValidationError,
};
