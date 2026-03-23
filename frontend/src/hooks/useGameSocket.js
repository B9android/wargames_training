/**
 * useGameSocket.js
 *
 * React hook that manages the WebSocket connection to the Python game server.
 *
 * Usage:
 *   const { send, lastMessage, readyState } = useGameSocket('ws://localhost:8765');
 */

import { useEffect, useRef, useState, useCallback } from 'react';

/** WebSocket ready-state constants (mirrors the browser WebSocket API). */
export const ReadyState = {
  CONNECTING: 0,
  OPEN: 1,
  CLOSING: 2,
  CLOSED: 3,
};

/**
 * @param {string} url  WebSocket server URL.
 * @returns {{ send: Function, lastMessage: object|null, readyState: number }}
 */
export default function useGameSocket(url) {
  const wsRef = useRef(/** @type {WebSocket|null} */ (null));
  const [lastMessage, setLastMessage] = useState(null);
  const [readyState, setReadyState] = useState(ReadyState.CONNECTING);

  useEffect(() => {
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => setReadyState(ReadyState.OPEN);
    ws.onclose = () => setReadyState(ReadyState.CLOSED);
    ws.onerror = () => setReadyState(ReadyState.CLOSED);
    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        setLastMessage(msg);
      } catch {
        // ignore malformed messages
      }
    };

    return () => {
      ws.close();
    };
  }, [url]);

  const send = useCallback((msg) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(typeof msg === 'string' ? msg : JSON.stringify(msg));
    }
  }, []);

  return { send, lastMessage, readyState };
}
