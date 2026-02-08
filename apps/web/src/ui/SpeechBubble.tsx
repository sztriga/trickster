import React, { useCallback, useEffect, useRef, useState } from "react";

export type BubblePosition = "ai" | "player";

export type SpeechBubbleHandle = {
  show: (text: string) => void;
  clear: () => void;
};

/**
 * Self-contained speech bubble with auto-dismiss animation.
 * Returns a handle to imperatively trigger bubbles plus the JSX element.
 */
export function useSpeechBubble(position: BubblePosition): [SpeechBubbleHandle, React.ReactNode] {
  const [text, setText] = useState<string | null>(null);
  const [visible, setVisible] = useState(false);
  const timer = useRef<number | null>(null);

  const show = useCallback((msg: string) => {
    if (timer.current) window.clearTimeout(timer.current);
    setText(msg);
    setVisible(true);
    timer.current = window.setTimeout(() => {
      setVisible(false);
      timer.current = window.setTimeout(() => {
        setText(null);
        timer.current = null;
      }, 400);
    }, 2000);
  }, []);

  const clear = useCallback(() => {
    if (timer.current) window.clearTimeout(timer.current);
    setText(null);
    setVisible(false);
    timer.current = null;
  }, []);

  // Cleanup on unmount
  useEffect(() => () => { if (timer.current) window.clearTimeout(timer.current); }, []);

  const animIn = position === "ai" ? "bubble-in-up" : "bubble-in-down";
  const animOut = position === "ai" ? "bubble-out-up" : "bubble-out-down";

  const node = text ? (
    <div className={`speech-bubble bubble-${position} ${visible ? animIn : animOut}`}>
      <span>{text}</span>
    </div>
  ) : null;

  return [{ show, clear }, node];
}
