import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from scipy.optimize import  differential_evolution
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.interpolate import griddata
import matplotlib.patches as mpatches
import sys
from os import listdir, system
sys.path.append(r'J:/Build/BiiGTKvs13/PythonTools')
import PythonTools
from PythonTools.registration import point_registration, registration_error, apply_registration

class RigidBody():

    def __init__(self, MarkerPoints, pivot = np.array([0.,0.,0.]), ExpFLE2 = 1.0, limitDistance = 1.0, gamma = 0.3, phi = 0.8):

        # Bounds
        self.LimitDistance = limitDistance
        self.gamma = gamma
        self.phi = phi

        # Markers placement
        self.Markers = MarkerPoints
        self.Pivot = pivot
        self.ExpFLE2 = ExpFLE2

        self.Update()



    def CalculateDkFk(self):
        # Number of markers
        MarkerNumber,c = self.Markers.shape
        self.MarkerNumber = MarkerNumber

        #for i in range(MarkerNumber):
        #    print "Marker # ", i, Markers[i,:]

        # Marker centroid
        self.MarkerCentroid = np.mean(self.Markers,axis=0)

        #print "Marker Centroid:" ,MarkerCentroid
        # Define PCA of markers
        Mcentered = self.Markers - np.matlib.repmat(self.MarkerCentroid,self.MarkerNumber,1)

        #for i in range(MarkerNumber):
        #    print "Marker Centered # ", i, Mcentered[i,:]

        cov_mat = np.cov(Mcentered.T)
        eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)

        # Reorder according to components
        eig_pairs_cov = [(np.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]
        eig_pairs_cov.sort(key=lambda x: x[0])
        eig_pairs_cov.reverse()

        self.PCA_axes = np.zeros((3,3))
        for i in range(3):
            self.PCA_axes[i,:] = eig_pairs_cov[i][1]

        #for i in range(3):
        #    print "PCA_axes # ", i, PCA_axes[i,:]

        # fk^2 is the mean of the squared distances of the markers to axis k
        self.fk = np.zeros((self.MarkerNumber,3))
        for marker in range(self.MarkerNumber):
            for kfk in range(3):
                self.fk[marker,kfk] = np.linalg.norm(np.cross(Mcentered[marker,:],self.PCA_axes[kfk,:]))/np.linalg.norm(self.PCA_axes[kfk,:])

        fk_out = np.zeros((3))
        fk_out[0] = np.sqrt(np.mean(self.fk[:,0]**2))
        fk_out[1] = np.sqrt(np.mean(self.fk[:,1]**2))
        fk_out[2] = np.sqrt(np.mean(self.fk[:,2]**2))

        self.fk2 = fk_out**2
        #for i in range(MarkerNumber):
        #    print "fk # ", i, fk[i,:]

        #for i in range(3):
        #    print "fk_out # ", i, fk_out[i]

        #print "Pivot Point centered\n", pivot_centered, "\n"
        self.PivotCentered = self.Pivot - self.MarkerCentroid
        self.dk = np.zeros((3))
        for kdk in range(0,3):
            self.dk[kdk] = np.linalg.norm(np.cross(self.PivotCentered,self.PCA_axes[kdk,:]))

        #print "Pivot ", pivot
        #print "PivotCentered ", PivotCentered
        #for i in range(3):
        #    print "dk # ", i, dk[i]

        self.dk2 = self.dk**2

    def EstimateTRE(self):
        # Calculate Dk and Fk
        self.CalculateDkFk()
        self.PPEfactor = 0.0
        for i in range(3):
            if self.fk2[i] == 0:
                self.PPEfactor = self.PPEfactor + 1.0
            else:
                self.PPEfactor = self.PPEfactor + self.dk2[i]/self.fk2[i]
        self.PPEfactor = self.PPEfactor/(3.0)

        self.ExpPPE2 = (self.ExpFLE2/self.MarkerNumber ) * (1 + self.PPEfactor)

    def CalculateTRE(self,point,Treg):
        pointHomogeneous = np.append(point, [1])
        pivotMoved = np.dot(Treg,pointHomogeneous)
        TRE = np.linalg.norm(pointHomogeneous[0:3] - pivotMoved[0:3])
        return TRE


    def Update(self):
        self.EstimateTRE()
        self.calculateDm()
        self.calculateSymmetry()

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

        ax.plot([self.Pivot[0]],[self.Pivot[1]],[self.Pivot[2]], 'o', markersize=8, color='blue', alpha=0.5)

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


        ax.set_xlabel('x', fontsize=20)
        ax.set_ylabel('y', fontsize=20)
        ax.set_zlabel('z', fontsize=20)

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

        red_patch = mpatches.Patch(color='red', label='Markers')
        blue_patch = mpatches.Patch(color='blue', label='Pivot Point')
        green_patch = mpatches.Patch(color='g', label='Marker\'s Centroid')
        plt.legend(handles=[red_patch, blue_patch, green_patch], fontsize=20, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title('Rigid Body and Principal Axis\n', fontsize=20)

        ax.axis('equal')

        plt.show()

    def SetVolumeBound(self,volume):
        self.Volume = volume

    def CostFunction(self,params):
        self.Markers = self.Volume.VolumeFunction(params)
        self.Update()
        return self.PPEfactor + self.ConstraintFunction()

    def ConstraintFunction(self):
        return self.gamma * (self.LimitDistance - self.Dm) + self.phi * (3*(1-2.0/self.MarkerNumber)*self.ExpFLE2 - self.symmetry)


    def permutations(self, iterable, r=None):
        # permutations('ABCD', 2) --> AB AC AD BA BC BD CA CB CD DA DB DC
        # permutations(range(3)) --> 012 021 102 120 201 210
        pool = tuple(iterable)
        n = len(pool)
        r = n if r is None else r
        if r > n:
            return
        indices = range(n)
        cycles = range(n, n-r, -1)
        yield tuple(pool[i] for i in indices[:r])
        while n:
            for i in reversed(range(r)):
                cycles[i] -= 1
                if cycles[i] == 0:
                    indices[i:] = indices[i+1:] + indices[i:i+1]
                    cycles[i] = n - i
                else:
                    j = cycles[i]
                    indices[i], indices[-j] = indices[-j], indices[i]
                    yield tuple(pool[i] for i in indices[:r])
                    break
            else:
                return

    def calculateSymmetry(self):
        # Create Permutations
        permutationsOptions =  self.permutations(range(self.MarkerNumber))
        minValueFRE = np.inf
        first = True
        for i in permutationsOptions:
            if first:
                first = False
            else:

                currentMarkers =  self.Markers[i,:]

                Treg = point_registration(self.Markers, currentMarkers)
                currentFRE = registration_error(Treg, self.Markers, currentMarkers)
                currentFRE = np.min(currentFRE**2)


                if currentFRE < minValueFRE:
                    minValueFRE = currentFRE

        self.symmetry = minValueFRE


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

    def optimizeTool(self, printResult = False, maxIt = 300,  gamma = 0.1, phi = 0.7):
        self.gamma = gamma
        self.phi = phi
        self.res = differential_evolution(self.CostFunction, bounds=self.Volume.Bounds, maxiter=maxIt, callback=None, disp=printResult,mutation =(0,1.99), tol = 0.0001)

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
        fig2 = plt.figure(figsize=(20,10))
        ax = fig.gca(projection='3d')
        ax2 = fig2.add_subplot(111)
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
            ax2.plot(np.linspace(-limit,limit,R),PPEsweep, label=str(i))

            ax.zaxis.set_major_locator(LinearLocator(10))
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

            ax.view_init(elev=40., azim=-20)

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
        surf = ax.plot_surface(grid_x,grid_y,grid_z, rstride=1, cstride=1, cmap=plt.get_cmap('brg'),linewidth=0, antialiased=False)

        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax.view_init(elev=40., azim=-20)
        plt.show()


        fig = plt.figure(figsize=(20,10))
        plt.imshow(grid_z.T, extent=(-10,10,-10,10), origin='lower',cmap=plt.get_cmap('brg'))

        plt.show()

    def ConvergenceStudy(self,trialsNumber = 10):
        Trials = trialsNumber

        ppeVector = np.zeros((Trials))
        solutions = np.zeros((Trials,self.Volume.ParamNumber))

        for t in range(Trials):
            self.optimizeTool(False,500)
            solutions[t,:] = self.res.x
            ppeVector[t] = self.ExpPPE2
            print 100*t/Trials

        self.convergencePPEVector = ppeVector
        self.convergenceSolutions = solutions

        #self.plotConvergenceResults()

    def plotConvergenceResults(self):
        fig = plt.figure(figsize=(10,10))
        plt.hist(self.convergencePPEVector, bins=25)
        plt.title('Convergence Study\n', fontsize=20)
        plt.xlabel('PPE value', fontsize=20)
        plt.ylabel('Number of cases', fontsize=20)
        plt.show()


        fig = plt.figure(figsize=(10,10))
        plt.boxplot(self.convergencePPEVector)
        plt.title('PPE Map Value\n', fontsize=20)
        plt.xlabel('Trial', fontsize=20)
        plt.ylabel('PPE value', fontsize=20)
        plt.show()

        print "Minimum value of Expected PPE: ", np.min(self.convergencePPEVector)
        print "Maximum value of Expected PPE: ", np.max(self.convergencePPEVector)
        print "Mean    value of Expected PPE: ", np.mean(self.convergencePPEVector)
        print "Median  value of Expected PPE: ", np.median(self.convergencePPEVector)
        print "Std     value of Expected PPE: ", np.std(self.convergencePPEVector)


    def SimulateTool(self, MeanFLEsquared, trials, Points, filePath):
        dimPoints = len(Points.shape)
        if(dimPoints == 2):
            NPoints, n = Points.shape # If there is more than one point, calculate shape and extract the number of points
        else:
            NPoints = 1
        print "Number of points: ", NPoints

        # Creation of the TRE simulations as the number of simulations
        TREVectorSimulations = np.zeros((trials,NPoints))
        TREVectorSimulationsExpected = np.zeros((trials,NPoints))
        TREVectorSimulationsGraph = np.zeros((trials,NPoints*2)) # Simulated and Expected

        # Extract the Optimal Marker Localization in 3D
        OptimalMarkers = myTool.Volume.VolumeFunction(optimalLocalization)

        # Creation of the FRE simulations as the number of simulations
        N, m = OptimalMarkers.shape
        FREVectorSimulationsEachMarker = np.zeros((trials,N))
        FREVectorSimulationsRMS = np.zeros((trials))
        FREVectorSimulationsRMSExpected = np.zeros((trials))

        # Creation of the Matrix Norm vector
        TregNormSimulations = np.zeros((trials))



        print "Optimal Markers: \n"
        N, m = OptimalMarkers.shape
        for i in range(N):
            print "Marker #", i, "  ", OptimalMarkers[i,:]

        # Simulations
        print "\n"
        print "Simulated Noise FLE std: ", MeanFLEsquared
        print "Number of trials: \n", trials

        myTool.ExpFLE2 = MeanFLEsquared
        for it in range(trials):

            # Create noise vector
            noiseMarkers = np.random.normal(loc=0.0, scale=MeanFLEsquared, size=OptimalMarkers.shape)

            # Add noise to the markers
            PerturbedMarkers = OptimalMarkers + noiseMarkers

            # Perform the registration of the points and Transform the pivot point
            Treg = point_registration(PerturbedMarkers, OptimalMarkers)
            TregNormSimulations[it] = np.linalg.norm(Treg)**2 / 4 # 4 is the size of the matrix for normalizing to the identity norm

            # Calculate the FRE for registration points

            FREVectorSimulationsEachMarker[it,:] = registration_error(Treg, PerturbedMarkers, OptimalMarkers)
            FREVectorSimulationsRMS[it] = np.mean(FREVectorSimulationsEachMarker[it,:]**2)
            FREVectorSimulationsRMSExpected[it] = (1-2.0/N) * MeanFLEsquared

            # Calculate the TRE at points
            for pt in range(NPoints):
                if(dimPoints == 2):
                    point = Points[pt,:]
                else:
                    point = Points
                TREVectorSimulations[it,pt] = myTool.CalculateTRE(point,Treg)**2
                TREVectorSimulationsGraph[it,pt*2] = TREVectorSimulations[it,pt]

                # Set the pivot
                myTool.Pivot = point
                # Update
                myTool.Update()
                # Extract the TRE
                TREVectorSimulationsExpected[it,pt] = myTool.ExpPPE2
                TREVectorSimulationsGraph[it,pt*2 + 1] = TREVectorSimulationsExpected[it,pt]


        MeanFLEsquaredVector = np.zeros((trials * NPoints)) + MeanFLEsquared

        Px = np.array([])
        Py = np.array([])
        Pz = np.array([])
        TrialVector = np.array([])

        TREVectorSimulationsVector = np.array([])
        TREVectorSimulationsExpectedVector = np.array([])
        FREVectorSimulationsRMSVector = np.array([])
        FREVectorSimulationsRMSExpectedVector = np.array([])

        for pt in range(NPoints):
                if(dimPoints == 2):
                    pass
                    Px = np.concatenate((Px,np.zeros((trials)) +  Points[pt,0]))
                    Py = np.concatenate((Py,np.zeros((trials)) +  Points[pt,1]))
                    Pz = np.concatenate((Pz,np.zeros((trials)) +  Points[pt,2]))
                    TrialVector = np.concatenate((TrialVector,np.array(range(trials))))

                    TREVectorSimulationsVector = np.concatenate((TREVectorSimulationsVector,TREVectorSimulations[:,pt]))
                    TREVectorSimulationsExpectedVector = np.concatenate((TREVectorSimulationsExpectedVector,TREVectorSimulationsExpected[:,pt]))
                    FREVectorSimulationsRMSVector = np.concatenate((FREVectorSimulationsRMSVector,FREVectorSimulationsRMS))
                    FREVectorSimulationsRMSExpectedVector = np.concatenate((FREVectorSimulationsRMSExpectedVector,FREVectorSimulationsRMSExpected))

                else:
                    point = Points
                    Px = Px + Points[0]
                    Py = Py + Points[1]
                    Pz = Pz + Points[2]
                    TrialVector = np.array(range(trials))
                    TREVectorSimulationsVector = TREVectorSimulations
                    TREVectorSimulationsExpectedVector = TREVectorSimulationsExpected
                    FREVectorSimulationsRMSVector = FREVectorSimulationsRMS
                    FREVectorSimulationsRMSExpectedVector = FREVectorSimulationsRMSExpected

        Data = np.transpose(np.vstack((TrialVector, MeanFLEsquaredVector,Px, Py, Pz, TREVectorSimulationsVector,TREVectorSimulationsExpectedVector,FREVectorSimulationsRMSVector, FREVectorSimulationsRMSExpectedVector)))
        np.savetxt(filePath, Data, fmt='%10.10f', delimiter=",", header="Trial, FLE rms, Px, Py, Pz, TRE rms, TRE rms Expected, FRE rms, FRE rms Expected")


class VolumeBound():

    def __init__(self, VolumeFunction, bounds):

        self.VolumeFunction = VolumeFunction
        self.Bounds = bounds

        # Number of parameters
        L = len(self.Bounds)

        self.ParamNumber = L



