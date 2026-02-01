# Portfolio Optimizer - Professional UI Design

## Design Specifications

### Color Palette
- **Background**: #0f1419 (charcoal)
- **Text Primary**: #ffffff (white)
- **Text Secondary**: rgba(255,255,255,0.55) (muted gray)
- **Borders**: rgba(255,255,255,0.08) (subtle gray)
- **Accent**: #4a90e2 (institutional blue)
- **No neons, no gradients**

### Typography
- **Font Family**: Inter (Google Fonts)
- **Title**: 2rem, weight 700
- **Section Headers**: 1.2rem, weight 700
- **Labels**: 0.75rem, uppercase, weight 600
- **Values**: 1.4rem, weight 700

### Layout Structure

#### 1. Header
```
Portfolio Optimizer
Modern Portfolio Theory · Risk Analytics · Efficient Frontier
```

#### 2. Sidebar - Control Panel (Grouped Sections)

**ASSETS**
- Ticker input (comma-separated)
- Period selector

**DATES**
- Start date
- End date

**OPTIMIZATION METHOD**
- Max Sharpe (radio)
- Minimum Variance (radio)
- Target Volatility (radio) + slider

**RISK-FREE ASSET**
- Enable/Disable toggle
- Risk-free rate input
- Leverage cap slider

**CONSTRAINTS**
- Long-only toggle
- Max weight per asset slider

**ENGINE**
- Use cvxpy toggle

#### 3. KPI Strip
- Expected Return (annual %)
- Volatility (annual %)
- Your Sharpe Ratio
- Max Sharpe (cloud)
- Selected Method

**Design**: Flat cards, thin borders, no shadows

#### 4. Main Content (Two Tabs)

**Tab 1: Portfolio Overview**
- Weights Table
- Allocation Chart
- Betas Table
- Covariance/Correlation Matrix (toggle)

**Tab 2: Frontier & Returns**
- Cumulative Returns Chart
- Efficient Frontier Plot (with markers)

#### 5. AI Assistant Panel
**Title**: Research Assistant
**Subtitle**: Explain portfolio, risk factors, correlations
- Quick prompts (buttons)
- Chat history
- Clean, academic tone
- No chatbot emojis

#### 6. Footer
```
© 2026 Portfolio Optimizer — For educational and analytical use
```

### Remove
- All emojis
- Bright colors
- Large icons
- Over-animation
- Marketing language

### Keep
- All core functionality
- Data visualization
- Academic tone
- Professional appearance
