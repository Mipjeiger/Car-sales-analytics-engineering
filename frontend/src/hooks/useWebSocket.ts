// frontend/src/hooks/useWebSocket.ts
import { useEffect, useRef, useState, useCallback } from 'react';
import { WS_URL, AUTH_TOKEN_KEY } from '@/utils/constants';

interface WebSocketMessage {
  type: string;
  [key: string]: any;
}

export function useWebSocket() {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const MAX_RECONNECT_ATTEMPTS = 5;

  const connect = useCallback(() => {
    const token = localStorage.getItem(AUTH_TOKEN_KEY);
    
    if (!token) {
      console.warn('No token available for WebSocket connection');
      setConnectionError('No authentication token available');
      return;
    }
    
    // Build WebSocket URL with token
    const wsUrl = `${WS_URL}?token=${encodeURIComponent(token)}`;
    console.log('🔄 Connecting to WebSocket:', wsUrl);
    
    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('✅ WebSocket connected successfully');
        setIsConnected(true);
        setConnectionError(null);
        reconnectAttemptsRef.current = 0; // Reset reconnect attempts on successful connection
        
        // Clear any reconnect timeout
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current);
          reconnectTimeoutRef.current = null;
        }

        // Send a welcome message to the server
        ws.send(JSON.stringify({
          type: 'ping',
          timestamp: new Date().toISOString()
        }));
      };

      ws.onclose = (event) => {
        console.log(`🔌 WebSocket disconnected: ${event.code} - ${event.reason || 'No reason provided'}`);
        setIsConnected(false);
        
        // Attempt to reconnect if not closed intentionally
        if (event.code !== 1000 && event.code !== 1001) {
          if (reconnectAttemptsRef.current < MAX_RECONNECT_ATTEMPTS) {
            reconnectAttemptsRef.current += 1;
            const delay = Math.min(5000 * reconnectAttemptsRef.current, 30000); // Exponential backoff
            console.log(`🔄 Attempting to reconnect (${reconnectAttemptsRef.current}/${MAX_RECONNECT_ATTEMPTS}) in ${delay}ms...`);
            
            reconnectTimeoutRef.current = setTimeout(() => {
              console.log('🔄 Reconnecting...');
              connect();
            }, delay);
          } else {
            console.error('❌ Max reconnection attempts reached');
            setConnectionError('Unable to reconnect to server. Please refresh the page.');
          }
        }
      };

      ws.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
        setConnectionError('WebSocket connection error');
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('📩 WebSocket message received:', data);
          
          // Handle specific message types
          if (data.type === 'connection') {
            console.log(`✅ Connected as user: ${data.user_id} (role: ${data.role})`);
          }
          
          if (data.type === 'error') {
            console.error('❌ Server error:', data.error);
            setConnectionError(data.error);
          }
          
          setLastMessage(data);
        } catch (error) {
          console.error('❌ Failed to parse WebSocket message:', error);
        }
      };
    } catch (error) {
      console.error('❌ Failed to create WebSocket connection:', error);
      setConnectionError('Failed to create WebSocket connection');
    }
  }, []);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close(1000, 'Intentional disconnect');
      wsRef.current = null;
    }
    setIsConnected(false);
    setConnectionError(null);
    reconnectAttemptsRef.current = 0;
  }, []);

  const sendMessage = useCallback((data: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
      return true;
    } else {
      console.warn('⚠️ WebSocket is not connected. ReadyState:', wsRef.current?.readyState);
      return false;
    }
  }, []);

  const sendChatMessage = useCallback((content: string) => {
    if (!content.trim()) {
      console.warn('⚠️ Cannot send empty message');
      return false;
    }
    
    return sendMessage({
      type: 'chat',
      content: content.trim(),
      timestamp: new Date().toISOString()
    });
  }, [sendMessage]);

  const sendPing = useCallback(() => {
    return sendMessage({
      type: 'ping',
      timestamp: new Date().toISOString()
    });
  }, [sendMessage]);

  const resetConnection = useCallback(() => {
    disconnect();
    setTimeout(() => {
      connect();
    }, 1000);
  }, [connect, disconnect]);

  // Connect on mount
  useEffect(() => {
    connect();
    
    // Cleanup on unmount
    return () => {
      disconnect();
    };
  }, []); // Empty dependency array

  // Optional: Auto-ping to keep connection alive
  useEffect(() => {
    if (!isConnected) return;
    
    const pingInterval = setInterval(() => {
      sendPing();
    }, 30000); // Send ping every 30 seconds
    
    return () => {
      clearInterval(pingInterval);
    };
  }, [isConnected, sendPing]);

  return {
    isConnected,
    lastMessage,
    connectionError,
    sendMessage,
    sendChatMessage,
    sendPing,
    disconnect,
    connect,
    resetConnection,
  };
}