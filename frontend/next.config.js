/** @type {import('next').NextConfig} */
const nextConfig = {
  transpilePackages: [
    'mapbox-gl',
    '@deck.gl/core',
    '@deck.gl/react',
    '@deck.gl/layers',
    '@deck.gl/geo-layers',
  ],
  webpack: (config) => {
    config.resolve.alias = {
      ...config.resolve.alias,
      'mapbox-gl': 'mapbox-gl/dist/mapbox-gl.js',
    };
    return config;
  },
};

module.exports = nextConfig;
