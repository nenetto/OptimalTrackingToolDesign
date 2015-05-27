import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from scipy.optimize import  differential_evolution
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.interpolate import griddata

class RigidBody():

    def __init__(self, MarkerPoints, pivot = [0.,0.,0.], ExpFLE2 = 1.0, limitDistance = 1.0, gamma = 0.3):

        # Bounds
        self.LimitDistance = limitDistance
        self.gamma = gamma

        # Markers placement
        self.Markers = MarkerPoints
        self.Pivot = pivot

        # Number of markers
        N,c = self.Markers.shape
        self.MarkerNumber = N

        # Marker centroid
        self.MarkerCentroid = np.mean(self.Markers,axis=0)

        # Define PCA of markers
        Mcentered = self.Markers - np.matlib.repmat(self.MarkerCentroid,self.MarkerNumber,1)
        cov_mat = np.cov(Mcentered.T)
        eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)

        # Reorder according to components
        eig_pairs_cov = [(np.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]
        eig_pairs_cov.sort(key=lambda x: x[0])
        eig_pairs_cov.reverse()

        self.PCA_axes = np.zeros((3,3))
        for i in range(3):
            self.PCA_axes[i,:] = eig_pairs_cov[i][1]

        # fk^2 is the mean of the squared distances of the markers to axis k
        self.fk = np.zeros((self.MarkerNumber,3))
        for marker in range(self.MarkerNumber):
            for kfk in range(3):
                self.fk[marker,kfk] = np.linalg.norm(np.cross(self.Markers[marker,:],self.PCA_axes[kfk,:]))/np.linalg.norm(self.PCA_axes[kfk,:])

        self.fk2 = np.mean(self.fk**2,axis = 0)


        #print "Pivot Point centered\n", pivot_centered, "\n"
        self.PivotCentered = self.Pivot - self.MarkerCentroid
        self.dk = np.zeros((3))
        for kdk in range(0,3):
            self.dk[kdk] = np.linalg.norm(np.cross(self.PivotCentered,self.PCA_axes[kdk,:]))

        self.dk2 = self.dk**2

        self.ExpFLE2 = ExpFLE2
        self.PPEfactor = 0.0
        for i in range(3):
            if self.fk2[i] == 0:
                self.PPEfactor = self.PPEfactor + 1.0
            else:
                self.PPEfactor = self.PPEfactor + self.dk2[i]/self.fk2[i]
        self.PPEfactor = self.PPEfactor/(3.0)

        self.ExpPPE2 = ( self.ExpFLE2/(self.MarkerNumber * 1.0) ) * (1 + self.PPEfactor)

        self.calculateDm()

    def Update(self):

        # Number of markers
        N,c = self.Markers.shape
        self.MarkerNumber = N

        # Marker centroid
        self.MarkerCentroid = np.mean(self.Markers,axis=0)

        # Define PCA of markers
        Mcentered = self.Markers - np.matlib.repmat(self.MarkerCentroid,self.MarkerNumber,1)
        cov_mat = np.cov(Mcentered.T)
        eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)

        # Reorder according to components
        eig_pairs_cov = [(np.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]
        eig_pairs_cov.sort(key=lambda x: x[0])
        eig_pairs_cov.reverse()

        self.PCA_axes = np.zeros((3,3))
        for i in range(3):
            self.PCA_axes[i,:] = eig_pairs_cov[i][1]

        # fk^2 is the mean of the squared distances of the markers to axis k
        self.fk = np.zeros((self.MarkerNumber,3))
        for marker in range(self.MarkerNumber):
            for kfk in range(3):
                self.fk[marker,kfk] = np.linalg.norm(np.cross(self.Markers[marker,:],self.PCA_axes[kfk,:]))/np.linalg.norm(self.PCA_axes[kfk,:])

        self.fk2 = np.mean(self.fk**2,axis = 0)


        #print "Pivot Point centered\n", pivot_centered, "\n"
        self.PivotCentered = self.Pivot - self.MarkerCentroid
        self.dk = np.zeros((3))
        for kdk in range(0,3):
            self.dk[kdk] = np.linalg.norm(np.cross(self.PivotCentered,self.PCA_axes[kdk,:]))

        self.dk2 = self.dk**2

        self.PPEfactor = 0.0
        for i in range(3):
            if self.fk2[i] == 0:
                self.PPEfactor = self.PPEfactor + 1.0
            else:
                self.PPEfactor = self.PPEfactor + self.dk2[i]/self.fk2[i]
        self.PPEfactor = self.PPEfactor/(3.0)

        self.ExpPPE2 = ( self.ExpFLE2/(self.MarkerNumber * 1.0) ) * (1 + self.PPEfactor)

        self.calculateDm()

    def __str__(self):
        print "Rigid Body:"
        print "\tNumber of Markers: ", self.MarkerNumber
        print "\tPivot Point      : ", self.Pivot
        print "\tCentroid         : ", self.MarkerCentroid
        print "\tPPE factor       : ", self.PPEfactor
        print "\tPPE Expected     : ", self.ExpPPE2
        print "\t-------------------------------"
        print "\tMarkers Localization:\n", self.Markers
        print "\t-------------------------------"
        return "\t"

    def plot(self):

        class Arrow3D(FancyArrowPatch):
            def __init__(self, xs, ys, zs, *args, **kwargs):
                FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
                self._verts3d = xs, ys, zs

            def draw(self, renderer):
                xs3d, ys3d, zs3d = self._verts3d
                xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
                self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
                FancyArrowPatch.draw(self, renderer)

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot Points
        ax.plot(self.Markers[:,0], self.Markers[:,1],self.Markers[:,2], 'o', markersize=8, color='red', alpha=0.5)
        for i in range(self.MarkerNumber):
            plt.plot([0,self.Markers[i,0]], [0,self.Markers[i,1]],[0,self.Markers[i,2]],'r:')

        # Plot Pivot Point
        ax.plot([0],[0],[0], 'o', markersize=8, color='blue', alpha=0.5)

        # Plot centroid
        ax.plot([self.MarkerCentroid[0]], [self.MarkerCentroid[1]], [self.MarkerCentroid[2]], 'o',markersize=10, color='green', alpha=0.5)

        # Plot restriction
        N = 100
        x = np.cos(np.deg2rad(np.linspace(0,360,N)))
        y = np.sin(np.deg2rad(np.linspace(0,360,N)))
        z = np.linspace(0,360,N)*0
        ax.plot(x, y, 'k:')

        # Plot PCA axes
        for v in self.PCA_axes:
            vc = v + self.MarkerCentroid
            a = Arrow3D([self.MarkerCentroid[0], vc[0]], [self.MarkerCentroid[1], vc[1]],[self.MarkerCentroid[2], vc[2]],\
                        mutation_scale=20, lw=3, arrowstyle="-|>", color="red", alpha = 0.2)
            ax.add_artist(a)


        ax.set_xlabel('x values')
        ax.set_ylabel('y values')
        ax.set_zlabel('z values')

        xlimVmin = np.min(self.Markers[:,0]) - np.abs(np.min(self.Markers[:,0])) * 0.10
        ylimVmin = np.min(self.Markers[:,1]) - np.abs(np.min(self.Markers[:,1])) * 0.10
        zlimVmin = np.min(self.Markers[:,2]) - np.abs(np.min(self.Markers[:,2])) * 0.10

        xlimVmax = np.max(self.Markers[:,0]) + np.abs(np.max(self.Markers[:,0])) * 0.10
        ylimVmax = np.max(self.Markers[:,1]) + np.abs(np.max(self.Markers[:,1])) * 0.10
        zlimVmax = np.max(self.Markers[:,2]) + np.abs(np.max(self.Markers[:,2])) * 0.10

        xc = np.array([xlimVmin,xlimVmax,vc[0]])
        yc = np.array([ylimVmin,ylimVmax,vc[1]])
        zc = np.array([zlimVmin,zlimVmax,vc[2]])

        xlimVmin = np.min(xc) - np.abs(np.min(xc)) * 0.10
        ylimVmin = np.min(yc) - np.abs(np.min(yc)) * 0.10
        zlimVmin = np.min(zc) - np.abs(np.min(zc)) * 0.10

        xlimVmax = np.max(xc) + np.abs(np.max(xc)) * 0.10
        ylimVmax = np.max(yc) + np.abs(np.max(yc)) * 0.10
        zlimVmax = np.max(zc) + np.abs(np.max(zc)) * 0.10


        ax.set_xlim([xlimVmin, xlimVmax])
        ax.set_ylim([ylimVmin, ylimVmax])
        ax.set_zlim([zlimVmin, zlimVmax])

        plt.title('Rigid Body\n')

        ax.axis('equal')

        plt.show()

    def SetVolumeBound(self,volume):
        self.Volume = volume

    def CostFunction(self,params):
        self.Markers = self.Volume.VolumeFunction(params)
        self.Update()
        return self.PPEfactor + self.ConstraintFunction()

    def ConstraintFunction(self):
        return self.gamma * np.abs(self.Dm - self.LimitDistance)

    def calculateDm(self):

        Dmcurrent = np.inf
        for i in range(self.MarkerNumber):
            for j in range(i+1,self.MarkerNumber):
                pi = self.Markers[i,:]
                pj = self.Markers[j,:]
                newDm = np.linalg.norm(pi-pj)
                if(newDm < Dmcurrent):
                    Dmcurrent = newDm

        self.Dm = Dmcurrent

    def optimizeTool(self, printResult = True, maxIt = 300):
        self.res = differential_evolution(self.CostFunction, bounds=self.Volume.Bounds, maxiter=maxIt, callback=None, disp=printResult,mutation =(0,1.99))

    def plotOptimalSurrounding(self,limit = 10, R = 50):
        for i in range(self.Volume.ParamNumber):
            fig = plt.figure(figsize=(5,5));

            sweepMarker = self.res.x[i] + np.linspace(-limit,limit,R)
            PPEsweep = np.zeros_like(sweepMarker)
            optimalParams = self.res.x
            currentParams = np.copy(optimalParams)

            for j in range(R):
                currentParams[i] = sweepMarker[j]
                a = self.CostFunction(currentParams)
                PPEsweep[j] = self.PPEfactor

            plt.plot(np.linspace(-limit,limit,R),PPEsweep)
            plt.title('ExpPPE2 if a param is moved')
            plt.show()

    def plotOptimalSurrounding3D(self,limit = 10, R = 50):


        angles = self.Volume.ParamNumber
        angles = np.linspace(0,360-360/angles,angles)

        fig = plt.figure(figsize=(20,10))
        ax = fig.gca(projection='3d')
        for i in range(self.Volume.ParamNumber):


            sweepMarker = self.res.x[i] + np.linspace(-limit,limit,R)
            PPEsweep = np.zeros_like(sweepMarker)
            optimalParams = self.res.x
            currentParams = np.copy(optimalParams)

            for j in range(R):
                currentParams[i] = sweepMarker[j]
                a = self.CostFunction(currentParams)
                PPEsweep[j] = self.PPEfactor

            xx = np.linspace(-limit,limit,R) * np.cos(np.deg2rad(angles[i]))
            yy = np.linspace(-limit,limit,R) * np.sin(np.deg2rad(angles[i]))
            zz = PPEsweep

            ax.plot(xx,yy,zz, '-', markersize=10, color='r')

            ax.zaxis.set_major_locator(LinearLocator(10))
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

            ax.view_init(elev=40., azim=45)

            if i > 0:
                X = np.concatenate((X,xx))
                Y = np.concatenate((Y,yy))
                Z = np.concatenate((Z,zz))
            else:
                X = xx
                Y = yy
                Z = zz

        plt.show()


        grid_x, grid_y = np.meshgrid(np.linspace(-limit,limit,R),np.linspace(-limit,limit,R))
        points = np.array([X,Y])
        grid_z = griddata(points.T, Z, (grid_x, grid_y), method='cubic')
        grid_z = np.nan_to_num(grid_z)

        fig = plt.figure(figsize=(20,10))
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(grid_x,grid_y,grid_z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'),linewidth=0, antialiased=False)

        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax.view_init(elev=45., azim=-20)
        plt.show()


        fig = plt.figure(figsize=(20,10))
        plt.imshow(grid_z.T, extent=(-10,10,-10,10), origin='lower',cmap=plt.get_cmap('rainbow'))

        plt.show()

    def ConvergenceStudy(self,trialsNumber = 10):
        Trials = trialsNumber

        ppeVector = np.zeros((Trials))
        solutions = np.zeros((Trials,self.Volume.ParamNumber))

        for t in range(Trials):
            self.optimizeTool(False,500)
            solutions[t,:] = self.res.x
            ppeVector[t] = self.ExpPPE2

        self.convergencePPEVector = ppeVector
        self.convergenceSolutions = solutions

        self.plotConvergenceResults()

    def plotConvergenceResults(self):
        fig = plt.figure(figsize=(20,10))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax2.boxplot(self.convergencePPEVector)
        ax1.hist(self.convergencePPEVector, bins=25)
        plt.show()

        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(111, projection='3d')
        # Plot Pivot Point
        ax.plot([0],[0],[0], 'o', markersize=8, color='blue', alpha=0.5)
        ax.set_xlabel('x values')
        ax.set_ylabel('y values')
        ax.set_zlabel('z values')
        ax.auto_scale_xyz([-5, 50], [-10, 10], [-5,5])
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.rainbow,
                linewidth=0.1, antialiased=False, alpha = 0.3)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.view_init(elev=40., azim=125)


        for i in range(Trials):
            M = self.Volume.VolumeFunction(self.convergenceSolutions[i,:])
            ax.plot(M[:,0], M[:,1],M[:,2], 'o', markersize=8, color='red', alpha=0.5)

        plt.show()

        print "Minimum value of Expected PPE: ", np.min(self.convergencePPEVector)
        print "Maximum value of Expected PPE: ", np.max(self.convergencePPEVector)
        print "Mean    value of Expected PPE: ", np.mean(self.convergencePPEVector)
        print "Median  value of Expected PPE: ", np.median(self.convergencePPEVector)
        print "Std     value of Expected PPE: ", np.std(self.convergencePPEVector)






class VolumeBound():

    def __init__(self, VolumeFunction, bounds):

        self.VolumeFunction = VolumeFunction
        self.Bounds = bounds

        # Number of parameters
        L = len(self.Bounds)

        self.ParamNumber = L



