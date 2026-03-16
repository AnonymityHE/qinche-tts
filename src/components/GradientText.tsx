import React from 'react'

interface GradientTextProps {
  children: React.ReactNode
  className?: string
  colors?: string[]
  animationSpeed?: number
}

const GradientText = React.memo(function GradientText({
  children,
  className = '',
  colors = ['#ffffff', '#a855f7', '#3b82f6', '#06b6d4', '#a855f7', '#ffffff'],
  animationSpeed = 10,
}: GradientTextProps) {
  const gradientStyle: React.CSSProperties = {
    backgroundImage: `linear-gradient(to right, ${colors.join(', ')})`,
    animationDuration: `${animationSpeed}s`,
  }
  return (
    <div className={`animated-gradient-text ${className}`}>
      <div className="text-content" style={gradientStyle}>
        {children}
      </div>
    </div>
  )
})

export default GradientText
