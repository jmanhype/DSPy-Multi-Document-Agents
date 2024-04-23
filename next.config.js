const withNextra = require('nextra')('nextra-theme-docs', './theme.config.js')

module.exports = withNextra({
  reactStrictMode: true,
  poweredByHeader: false,
  async rewrites() {
    return [
      // Define custom rewrites or redirects
    ]
  },
  async headers() {
    return [
      // Define security headers or other response headers
      {
        source: '/:path*',
        headers: [
          { key: 'X-Content-Type-Options', value: 'nosniff' },
          { key: 'X-Frame-Options', value: 'DENY' },
        ],
      },
    ]
  },
  // other custom Next.js configurations
});
