/**
 * ConnectionLines
 * 在三栏之间渲染 SVG 贝塞尔连线
 *
 * 连线颜色规则：
 *   Session ↔ Effective : 橙黄  #f59e0b
 *   Effective ↔ Longterm: 紫色  #8b5cf6
 *
 * 连线从卡片右边中点 → 对侧卡片左边中点，画 S 型贝塞尔曲线。
 * 卡片用 data-card-id="<id>" 属性标记位置。
 */
import { useEffect, useRef, useState, useCallback } from 'react';
import type { SelectionState } from '../types';

interface Line {
  x1: number; y1: number;
  x2: number; y2: number;
  color: string;
  key: string;
}

interface Props {
  selection: SelectionState;
  containerRef: React.RefObject<HTMLElement | null>;
}

function getCardEl(id: string): Element | null {
  return document.querySelector(`[data-card-id="${CSS.escape(id)}"]`);
}

/** 返回元素右边中点（相对于 container 左上角） */
function rightMid(el: Element, container: DOMRect): { x: number; y: number } {
  const r = el.getBoundingClientRect();
  return {
    x: r.right - container.left,
    y: r.top + r.height / 2 - container.top,
  };
}

/** 返回元素左边中点（相对于 container 左上角） */
function leftMid(el: Element, container: DOMRect): { x: number; y: number } {
  const r = el.getBoundingClientRect();
  return {
    x: r.left - container.left,
    y: r.top + r.height / 2 - container.top,
  };
}

/** 只连接视口内可见的元素（避免画到屏幕外的线） */
function isVisible(el: Element, container: DOMRect): boolean {
  const r = el.getBoundingClientRect();
  return r.bottom > container.top && r.top < container.bottom &&
         r.right > container.left && r.left < container.right;
}

function buildLines(selection: SelectionState, containerRect: DOMRect): Line[] {
  const lines: Line[] = [];

  if (!selection.id) return lines;

  const sessionIds = Array.from(selection.highlightedSessionIds);
  const effectiveIds = Array.from(selection.highlightedEffectiveIds);
  const longtermIds = Array.from(selection.highlightedLongtermIds);

  // Session → Effective 连线（橙黄）
  for (const sid of sessionIds) {
    const sEl = getCardEl(sid);
    if (!sEl || !isVisible(sEl, containerRect)) continue;
    const sRight = rightMid(sEl, containerRect);

    for (const eid of effectiveIds) {
      const eEl = getCardEl(eid);
      if (!eEl || !isVisible(eEl, containerRect)) continue;
      const eLeft = leftMid(eEl, containerRect);

      lines.push({
        x1: sRight.x, y1: sRight.y,
        x2: eLeft.x, y2: eLeft.y,
        color: '#f59e0b',
        key: `s-e-${sid}-${eid}`,
      });
    }
  }

  // Effective → Longterm 连线（紫色）
  for (const eid of effectiveIds) {
    const eEl = getCardEl(eid);
    if (!eEl || !isVisible(eEl, containerRect)) continue;
    const eRight = rightMid(eEl, containerRect);

    for (const lid of longtermIds) {
      const lEl = getCardEl(lid);
      if (!lEl || !isVisible(lEl, containerRect)) continue;
      const lLeft = leftMid(lEl, containerRect);

      lines.push({
        x1: eRight.x, y1: eRight.y,
        x2: lLeft.x, y2: lLeft.y,
        color: '#8b5cf6',
        key: `e-l-${eid}-${lid}`,
      });
    }
  }

  return lines;
}

function BezierPath({ x1, y1, x2, y2, color }: Omit<Line, 'key'>) {
  const dx = Math.abs(x2 - x1) * 0.45;
  const d = `M ${x1},${y1} C ${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`;
  return (
    <>
      {/* 白色光晕让线在暗背景上更清晰 */}
      <path d={d} stroke="rgba(255,255,255,0.08)" strokeWidth={4} fill="none" strokeLinecap="round" />
      <path d={d} stroke={color} strokeWidth={1.8} fill="none" strokeLinecap="round"
            strokeDasharray="0"
            style={{ filter: `drop-shadow(0 0 4px ${color}99)` }} />
      {/* 终点小圆点 */}
      <circle cx={x2} cy={y2} r={3.5} fill={color} style={{ filter: `drop-shadow(0 0 3px ${color})` }} />
      <circle cx={x1} cy={y1} r={3.5} fill={color} style={{ filter: `drop-shadow(0 0 3px ${color})` }} />
    </>
  );
}

export function ConnectionLines({ selection, containerRef }: Props) {
  const [lines, setLines] = useState<Line[]>([]);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const rafRef = useRef<number | null>(null);

  const recompute = useCallback(() => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    setSize({ w: rect.width, h: rect.height });
    setLines(buildLines(selection, rect));
  }, [selection, containerRef]);

  // 每次 selection 变化后，等 DOM 渲染+滚动完成再计算
  useEffect(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    // 两帧延迟确保卡片滚动/渲染完毕
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = requestAnimationFrame(recompute);
    });
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, [recompute]);

  // 监听窗口 resize 和容器内滚动
  useEffect(() => {
    const handler = () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      rafRef.current = requestAnimationFrame(recompute);
    };
    window.addEventListener('resize', handler);
    // 监听所有 .panel-body 滚动
    const scrollEls = document.querySelectorAll('.panel-body');
    scrollEls.forEach(el => el.addEventListener('scroll', handler));
    return () => {
      window.removeEventListener('resize', handler);
      scrollEls.forEach(el => el.removeEventListener('scroll', handler));
    };
  }, [recompute]);

  if (!lines.length) return null;

  return (
    <svg
      className="connection-svg"
      width={size.w}
      height={size.h}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        pointerEvents: 'none',
        zIndex: 10,
        overflow: 'visible',
      }}
    >
      {lines.map(l => (
        <BezierPath key={l.key} x1={l.x1} y1={l.y1} x2={l.x2} y2={l.y2} color={l.color} />
      ))}
    </svg>
  );
}

