/**
 * Performance Hooks - 2026 Mobile Optimization Standards
 * Based on web.dev, MDN, and industry best practices
 */

import { useEffect, useRef, useState, useCallback } from 'react'

/**
 * Debounce - For input events (search, resize)
 * Only executes after user stops for `delay` ms
 */
export function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value)

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedValue(value), delay)
    return () => clearTimeout(timer)
  }, [value, delay])

  return debouncedValue
}

/**
 * Debounced callback version
 */
export function useDebouncedCallback<T extends (...args: any[]) => any>(
  callback: T,
  delay: number
): (...args: Parameters<T>) => void {
  const timeoutRef = useRef<NodeJS.Timeout | null>(null)

  return useCallback(
    (...args: Parameters<T>) => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current)
      timeoutRef.current = setTimeout(() => callback(...args), delay)
    },
    [callback, delay]
  )
}

/**
 * Throttle - For scroll/resize events
 * Limits execution to once per `limit` ms
 */
export function useThrottle<T>(value: T, limit: number): T {
  const [throttledValue, setThrottledValue] = useState<T>(value)
  const lastRan = useRef(Date.now())

  useEffect(() => {
    const handler = setTimeout(() => {
      if (Date.now() - lastRan.current >= limit) {
        setThrottledValue(value)
        lastRan.current = Date.now()
      }
    }, limit - (Date.now() - lastRan.current))

    return () => clearTimeout(handler)
  }, [value, limit])

  return throttledValue
}

/**
 * Throttled callback version
 */
export function useThrottledCallback<T extends (...args: any[]) => any>(
  callback: T,
  limit: number
): (...args: Parameters<T>) => void {
  const inThrottle = useRef(false)

  return useCallback(
    (...args: Parameters<T>) => {
      if (!inThrottle.current) {
        callback(...args)
        inThrottle.current = true
        setTimeout(() => (inThrottle.current = false), limit)
      }
    },
    [callback, limit]
  )
}

/**
 * Intersection Observer - For lazy loading & animations
 * Triggers callback when element enters viewport
 */
export function useIntersectionObserver(
  options: IntersectionObserverInit = {}
): [React.RefObject<HTMLElement>, boolean] {
  const ref = useRef<HTMLElement>(null)
  const [isIntersecting, setIsIntersecting] = useState(false)

  useEffect(() => {
    const element = ref.current
    if (!element) return

    const observer = new IntersectionObserver(
      ([entry]) => {
        setIsIntersecting(entry.isIntersecting)
      },
      {
        threshold: 0.1,
        rootMargin: '50px',
        ...options,
      }
    )

    observer.observe(element)
    return () => observer.disconnect()
  }, [options])

  return [ref, isIntersecting]
}

/**
 * Lazy load images with Intersection Observer
 */
export function useLazyImage(src: string): [React.RefObject<HTMLImageElement>, string, boolean] {
  const ref = useRef<HTMLImageElement>(null)
  const [loadedSrc, setLoadedSrc] = useState('')
  const [isLoaded, setIsLoaded] = useState(false)

  useEffect(() => {
    const element = ref.current
    if (!element) return

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setLoadedSrc(src)
          observer.disconnect()
        }
      },
      { rootMargin: '200px' }
    )

    observer.observe(element)
    return () => observer.disconnect()
  }, [src])

  useEffect(() => {
    if (loadedSrc) {
      const img = new Image()
      img.onload = () => setIsLoaded(true)
      img.src = loadedSrc
    }
  }, [loadedSrc])

  return [ref, loadedSrc, isLoaded]
}

/**
 * Request Animation Frame - For smooth animations
 * Syncs with browser's refresh rate
 */
export function useAnimationFrame(callback: (deltaTime: number) => void, isRunning = true) {
  const requestRef = useRef<number>()
  const previousTimeRef = useRef<number>()

  useEffect(() => {
    if (!isRunning) return

    const animate = (time: number) => {
      if (previousTimeRef.current !== undefined) {
        const deltaTime = time - previousTimeRef.current
        callback(deltaTime)
      }
      previousTimeRef.current = time
      requestRef.current = requestAnimationFrame(animate)
    }

    requestRef.current = requestAnimationFrame(animate)
    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current)
    }
  }, [callback, isRunning])
}

/**
 * Media Query Hook - Responsive JS logic
 */
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(false)

  useEffect(() => {
    const media = window.matchMedia(query)
    setMatches(media.matches)

    const listener = (e: MediaQueryListEvent) => setMatches(e.matches)
    media.addEventListener('change', listener)
    return () => media.removeEventListener('change', listener)
  }, [query])

  return matches
}

/**
 * Touch device detection
 */
export function useIsTouchDevice(): boolean {
  const [isTouch, setIsTouch] = useState(false)

  useEffect(() => {
    setIsTouch(
      'ontouchstart' in window ||
      navigator.maxTouchPoints > 0 ||
      window.matchMedia('(pointer: coarse)').matches
    )
  }, [])

  return isTouch
}

/**
 * Scroll position with throttling
 */
export function useScrollPosition(throttleMs = 100): { x: number; y: number } {
  const [position, setPosition] = useState({ x: 0, y: 0 })

  useEffect(() => {
    let ticking = false

    const handleScroll = () => {
      if (!ticking) {
        window.requestAnimationFrame(() => {
          setPosition({
            x: window.scrollX,
            y: window.scrollY,
          })
          ticking = false
        })
        ticking = true
      }
    }

    window.addEventListener('scroll', handleScroll, { passive: true })
    return () => window.removeEventListener('scroll', handleScroll)
  }, [throttleMs])

  return position
}

/**
 * Reduced motion preference
 */
export function usePrefersReducedMotion(): boolean {
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(false)

  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)')
    setPrefersReducedMotion(mediaQuery.matches)

    const listener = (e: MediaQueryListEvent) => setPrefersReducedMotion(e.matches)
    mediaQuery.addEventListener('change', listener)
    return () => mediaQuery.removeEventListener('change', listener)
  }, [])

  return prefersReducedMotion
}

/**
 * Image preloader - Preload critical images
 */
export function useImagePreloader(urls: string[]): boolean {
  const [loaded, setLoaded] = useState(false)

  useEffect(() => {
    let isMounted = true
    const images: HTMLImageElement[] = []

    Promise.all(
      urls.map(
        (url) =>
          new Promise<void>((resolve) => {
            const img = new Image()
            images.push(img)
            img.onload = () => resolve()
            img.onerror = () => resolve() // Don't fail on error
            img.src = url
          })
      )
    ).then(() => {
      if (isMounted) setLoaded(true)
    })

    return () => {
      isMounted = false
      images.forEach((img) => (img.src = ''))
    }
  }, [urls])

  return loaded
}

/**
 * Online/Offline status
 */
export function useOnlineStatus(): boolean {
  const [isOnline, setIsOnline] = useState(true)

  useEffect(() => {
    setIsOnline(navigator.onLine)

    const handleOnline = () => setIsOnline(true)
    const handleOffline = () => setIsOnline(false)

    window.addEventListener('online', handleOnline)
    window.addEventListener('offline', handleOffline)

    return () => {
      window.removeEventListener('online', handleOnline)
      window.removeEventListener('offline', handleOffline)
    }
  }, [])

  return isOnline
}
